class CustomDataset(Dataset):
    def __init__(self, df, path_col,  mode='train'):
        self.df = df
        self.path_col = path_col
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'val':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'inference':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            data = {
                'image':image,
            }
            return data

    def train_transform(self, image):
        pass



class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.clf = nn.Sequential(
            nn.Tanh(),
            nn.LazyLinear(25),
        )

#     @torch.compile
    def forward(self, x, label=None):
        x = self.model(x).pooler_output
        x = self.clf(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(x, label)
        probs = nn.LogSoftmax(dim=-1)(x)
        return probs, loss

class LitCustomModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = CustomModel(model)
        self.validation_step_output = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt

    def training_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.log(f"train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.validation_step_output.append([probs,label])
        return loss

    def predict_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        probs, _ = self.model(x)
        return probs

    def validation_epoch_end(self, step_output):
        pred = torch.cat([x for x, _ in self.validation_step_output]).cpu().detach().numpy().argmax(1)
        label = torch.cat([label for _, label in self.validation_step_output]).cpu().detach().numpy()
        score = f1_score(label,pred, average='macro')
        self.log("val_score", score)
        self.validation_step_output.clear()
        return score



skf = StratifiedKFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)



for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, train_df['class'])):
    train_fold_df = train_df.loc[train_index,:]
    val_fold_df = train_df.loc[val_index,:]

    train_dataset = CustomDataset(train_fold_df, 'img_path', mode='train')
    val_dataset = CustomDataset(val_fold_df, 'img_path', mode='val')

    train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=BATCH_SIZE*2)

    model = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")
    lit_model = LitCustomModel(model)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_score',
        mode='max',
        dirpath='./checkpoints/',
        filename=f'swinv2-large-resize-fold_idx={fold_idx}'+'-{epoch:02d}-{train_loss:.4f}-{val_score:.4f}',
        save_top_k=1,
        save_weights_only=True,
        verbose=True
    )
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=3)
    trainer = L.Trainer(max_epochs=100, accelerator='auto', precision=32, callbacks=[checkpoint_callback, earlystopping_callback], val_check_interval=0.5)
    trainer.fit(lit_model, train_dataloader, val_dataloader)

    model.cpu()
    lit_model.cpu()
    del model, lit_model, checkpoint_callback, earlystopping_callback, trainer
    gc.collect()
    torch.cuda.empty_cache()


fold_preds = []
for checkpoint_path in glob('./checkpoints/swinv2-large-resize*.ckpt'):
    model = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")
    lit_model = LitCustomModel.load_from_checkpoint(checkpoint_path, model=model)
    trainer = L.Trainer( accelerator='auto', precision=32)
    preds = trainer.predict(lit_model, test_dataloader)
    preds = torch.cat(preds,dim=0).detach().cpu().numpy().argmax(1)
    fold_preds.append(preds)
pred_ensemble = list(map(lambda x: np.bincount(x).argmax(),np.stack(fold_preds,axis=1)))
