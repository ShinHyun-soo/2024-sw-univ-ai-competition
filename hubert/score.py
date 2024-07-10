import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import pytorch_lightning as pl

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import HubertForSequenceClassification, AutoFeatureExtractor, AutoConfig

import bitsandbytes as bnb

# Constants
DATA_DIR = ''  # Adjust this path as necessary
PREPROC_DIR = './preproc'
SUBMISSION_DIR = './submission'
MODEL_DIR = './model'
SAMPLING_RATE = 16000
SEED = 42
N_FOLD = 20
BATCH_SIZE = 2
NUM_LABELS = 2
AUDIO_MODEL_NAME = 'abhishtagatya/hubert-base-960h-itw-deepfake'


# Utility functions
def accuracy(preds, labels):
    return (preds == labels).float().mean()

def getAudios(df):
    audios = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio, _ = librosa.load(row['path'], sr=SAMPLING_RATE)
            audios.append(audio)
            valid_indices.append(idx)
        except FileNotFoundError:
            print(f"File not found: {row['path']}. Skipping.")
        except Exception as e:
            print(f"Error loading {row['path']}: {e}. Skipping.")
    return audios, valid_indices


# Dataset class
class MyDataset(Dataset):
    def __init__(self, audio, audio_feature_extractor, labels=None):
        if labels is None:
            labels = [[0] * NUM_LABELS for _ in range(len(audio))]
        self.labels = np.array(labels).astype(np.float32)
        self.audio = audio
        self.audio_feature_extractor = audio_feature_extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audio[idx]
        audio_feature = self.audio_feature_extractor(raw_speech=audio, return_tensors='np', sampling_rate=SAMPLING_RATE)
        audio_values, audio_attn_mask = audio_feature['input_values'][0], audio_feature['attention_mask'][0]

        item = {
            'label': label,
            'audio_values': audio_values,
            'audio_attn_mask': audio_attn_mask,
        }

        return item


# Collate function
def collate_fn(samples):
    batch_labels = []
    batch_audio_values = []
    batch_audio_attn_masks = []

    for sample in samples:
        batch_labels.append(sample['label'])
        batch_audio_values.append(torch.tensor(sample['audio_values']))
        batch_audio_attn_masks.append(torch.tensor(sample['audio_attn_mask']))

    batch_labels = torch.tensor(batch_labels)
    batch_audio_values = pad_sequence(batch_audio_values, batch_first=True)
    batch_audio_attn_masks = pad_sequence(batch_audio_attn_masks, batch_first=True)

    batch = {
        'label': batch_labels,
        'audio_values': batch_audio_values,
        'audio_attn_mask': batch_audio_attn_masks,
    }

    return batch


# Lightning Model class
class MyLitModel(pl.LightningModule):
    def __init__(self, audio_model_name, num_labels, n_layers=1, projector=True, classifier=True, dropout=0.07,
                 lr_decay=1):
        super(MyLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(audio_model_name, num_labels=num_labels)
        self.config.activation_dropout = dropout
        self.config.attention_dropout = dropout
        self.config.final_dropout = dropout
        self.config.hidden_dropout = dropout
        self.config.hidden_dropout_prob = dropout
        self.audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, config=self.config)
        self.lr_decay = lr_decay
        self._do_reinit(n_layers, projector, classifier)

    def forward(self, audio_values, audio_attn_mask):
        logits = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask).logits
        return logits

    def training_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        labels = batch['label']

        logits = self(audio_values, audio_attn_mask)
        loss = nn.BCEWithLogitsLoss()(logits, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        labels = batch['label']

        logits = self(audio_values, audio_attn_mask)
        loss = nn.BCEWithLogitsLoss()(logits, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']

        logits = self(audio_values, audio_attn_mask)
        probs = torch.sigmoid(logits)

        return probs

    def configure_optimizers(self):
        lr = 1e-5
        layer_decay = self.lr_decay
        weight_decay = 0.01
        llrd_params = self._get_llrd_params(lr=lr, layer_decay=layer_decay, weight_decay=weight_decay)
        optimizer = bnb.optim.Adam8bit(llrd_params)  # optimizer 을 8bit 로 하여 계산 속도 향상 및 vram 사용량 감축
        return optimizer

    def _get_llrd_params(self, lr, layer_decay, weight_decay):
        n_layers = self.audio_model.config.num_hidden_layers
        llrd_params = []
        for name, value in list(self.named_parameters()):
            if ('bias' in name) or ('layer_norm' in name):
                llrd_params.append({"params": value, "lr": lr, "weight_decay": 0.0})
            elif ('emb' in name) or ('feature' in name):
                llrd_params.append(
                    {"params": value, "lr": lr * (layer_decay ** (n_layers + 1)), "weight_decay": weight_decay})
            elif 'encoder.layer' in name:
                for n_layer in range(n_layers):
                    if f'encoder.layer.{n_layer}' in name:
                        llrd_params.append(
                            {"params": value, "lr": lr * (layer_decay ** (n_layer + 1)), "weight_decay": weight_decay})
            else:
                llrd_params.append({"params": value, "lr": lr, "weight_decay": weight_decay})
        return llrd_params

    def _do_reinit(self, n_layers=0, projector=True, classifier=True):
        if projector:
            self.audio_model.projector.apply(self._init_weight_and_bias)
        if classifier:
            self.audio_model.classifier.apply(self._init_weight_and_bias)

        for n in range(n_layers):
            self.audio_model.hubert.encoder.layers[-(n + 1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.audio_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Main script
if __name__ == '__main__':
    seed_everything(SEED)

    # 사운드 특징 추출
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
    audio_feature_extractor.return_attention_mask = True

    # 데이터 로드
    train_df = pd.read_csv('./combined_train_aug_fixed_1.csv')
    print(f"Train DataFrame shape: {train_df.shape}")
    test_df = pd.read_csv('./test.csv')
    print(f"Test DataFrame shape: {test_df.shape}")
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_DIR, x))
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_DIR, x))

    # 싱글 라벨을 멀티 라벨로 변환
    train_df['label'] = train_df['label'].apply(
        lambda x: [1, 0] if x in [0, 4] else (
            [0, 1] if x in [1, 5] else (
                [1, 1] if x == 2 else [0, 0]
            )
        )
    )
    test_df['label'] = [[0, 0]] * len(test_df)

    print(f"Processed Train DataFrame head: {train_df.head()}")
    print(f"Processed Test DataFrame head: {test_df.head()}")

    train_audios, valid_indices = getAudios(train_df)
    print(f"Number of valid train audios: {len(train_audios)}")
    train_df = train_df.iloc[valid_indices].reset_index(drop=True)
    train_labels = np.array(train_df['label'].tolist())

    # K 폴드
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    for fold_idx, (train_indices, val_indices) in enumerate(
            skf.split(train_labels, train_labels.argmax(axis=1))):
        print(
            f"Fold {fold_idx}: Train indices length: {len(train_indices)}, Validation indices length: {len(val_indices)}")
        train_fold_audios = [train_audios[train_index] for train_index in train_indices]
        val_fold_audios = [train_audios[val_index] for val_index in val_indices]

        train_fold_labels = train_labels[train_indices]
        val_fold_labels = train_labels[val_indices]
        train_fold_ds = MyDataset(train_fold_audios, audio_feature_extractor, train_fold_labels)
        val_fold_ds = MyDataset(val_fold_audios, audio_feature_extractor, val_fold_labels)
        train_fold_dl = DataLoader(train_fold_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        val_fold_dl = DataLoader(val_fold_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        checkpoint_acc_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=MODEL_DIR,
            filename=f'fold_{fold_idx}' + '_{epoch:02d}-{val_loss:.4f}-{train_loss:.4f}',
            save_top_k=30,
            mode='max'
        )

        my_lit_model = MyLitModel(
            audio_model_name=AUDIO_MODEL_NAME,
            num_labels=NUM_LABELS,
            n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=0.8
        )

        trainer = pl.Trainer(
            accelerator='cuda',
            max_epochs=1,
            precision='16-mixed',
            val_check_interval=0.1,
            callbacks=[checkpoint_acc_callback],
            accumulate_grad_batches=2
            # batch_size * accumulate_grad_batches = 가 실질적인 배치 사이즈임. (vram 은 batch_size 기준으로 소모함.)
        )

        print(f"Starting training for fold {fold_idx}...")
        trainer.fit(my_lit_model, train_fold_dl, val_fold_dl)
        print(f"Training completed for fold {fold_idx}.")

        del my_lit_model

    # 테스트 셋 예측
    test_audios, _ = getAudios(test_df)
    print(f"Number of test audios: {len(test_audios)}")
    test_ds = MyDataset(test_audios, audio_feature_extractor)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    pretrained_models = list(map(lambda x: os.path.join(MODEL_DIR, x), os.listdir(MODEL_DIR)))

    test_preds = []
    trainer = pl.Trainer(
        accelerator='cuda',
        precision=16,
    )

    for pretrained_model_path in pretrained_models:
        pretrained_model = MyLitModel.load_from_checkpoint(
            pretrained_model_path,
            audio_model_name=AUDIO_MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        test_pred = trainer.predict(pretrained_model, test_dl)
        test_pred = torch.cat(test_pred).detach().cpu().numpy()
        test_preds.append(test_pred)
        del pretrained_model

    # preds 를 vstack 으로 나열?
    test_preds = np.vstack(test_preds)
    # 0열 값을 fake, 1열 값을 real
    submission_df = pd.read_csv(os.path.join('sample_submission.csv'))
    submission_df['fake'] = test_preds[:, 0]
    submission_df['real'] = test_preds[:, 1]
    submission_df.to_csv(os.path.join(SUBMISSION_DIR, '20_fold212.csv'), index=False)
