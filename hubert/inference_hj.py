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
from pytorch_lightning import seed_everything
from transformers import HubertForSequenceClassification, AutoFeatureExtractor, AutoConfig
from torch.optim import AdamW
import bitsandbytes as bnb

# Constants
DATA_DIR = ''  # Adjust this path as necessary
PREPROC_DIR = './preproc'
SUBMISSION_DIR = './submission'
MODEL_DIR = './model'
SAMPLING_RATE = 16000
SEED = 42
N_FOLD = 10
BATCH_SIZE = 2
NUM_LABELS = 2
AUDIO_MODEL_NAME = 'abhishtagatya/hubert-base-960h-asv19-deepfake'

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


def collate_fn(samples):
    batch_labels = []
    batch_audio_values = []
    batch_audio_attn_masks = []

    for sample in samples:
        batch_labels.append(sample['label'])
        batch_audio_values.append(torch.tensor(sample['audio_values']))
        batch_audio_attn_masks.append(torch.tensor(sample['audio_attn_mask']))

    batch_labels = np.array(batch_labels)
    batch_labels = torch.tensor(batch_labels)
    batch_audio_values = pad_sequence(batch_audio_values, batch_first=True)
    batch_audio_attn_masks = pad_sequence(batch_audio_attn_masks, batch_first=True)

    batch = {
        'label': batch_labels,
        'audio_values': batch_audio_values,
        'audio_attn_mask': batch_audio_attn_masks,
    }

    return batch

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
        optimizer = bnb.optim.AdamW(llrd_params)  # optimizer 을 8bit 로 하여 계산 속도 향상 및 vram 사용량 감축
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


seed_everything(SEED)

# 사운드 특징 추출
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
audio_feature_extractor.return_attention_mask = True


test_df = pd.read_csv('./test.csv')
test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_DIR, x))


test_df['label'] = [[0, 0]] * len(test_df)


# 테스트 셋 예측
test_audios, _ = getAudios(test_df)


test_ds = MyDataset(test_audios, audio_feature_extractor)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)


pretrained_models = list(map(lambda x: os.path.join(MODEL_DIR, x), os.listdir(MODEL_DIR)))

test_preds = []
trainer = pl.Trainer(
    accelerator='cuda',
    precision='16',
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

# preds 를 vstack 으로 행 변환 reshape 느낌
test_preds = np.array(test_preds)
mean_preds = np.mean(test_preds, axis=0)
# 0열 값을 fake, 1열 값을 real
submission_df = pd.read_csv(os.path.join('sample_submission.csv'))
submission_df['fake'] = mean_preds[:, 0]
submission_df['real'] = mean_preds[:, 1]
submission_df.to_csv(os.path.join(SUBMISSION_DIR, 'zesus_plz_3.csv'), index=False)