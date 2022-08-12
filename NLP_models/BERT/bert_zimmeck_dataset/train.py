import pandas as pd
import numpy as np
import os

import tensorflow

import torch
import torch.nn as nn

import pytorch_lightning as pl

import torchmetrics

from torch.utils.data import Dataset, DataLoader


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertModel,BertConfig


class CFG:
    seed = 42
    model_name = 'bert-base-uncased'
    pretrained = True
    num_classes = 54
    lr = 5e-3
    max_lr = 1e-3
    pct_start = 0.2
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 40
    batch_size = 8
    accum=1
    n_fold = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = ""
    filename = 'zimmeck_dataset_filtered.csv'
    embed_dim=768
    hidden_dim=768*4
    DEBUG = False

seed_everything(CFG.seed)


# read data
df_all = pd.read_csv(os.path.join(CFG.base_dir,CFG.filename))
df_all.dropna(inplace=True)

if CFG.DEBUG == True:
    df_all = df_all[:200]
    CFG.num_epochs = 10

# k-fold
skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

for train_idx, valid_idx in skf.split(df_all['review'], df_all["label"]):
    df_train = df_all.iloc[train_idx]
    df_valid = df_all.iloc[valid_idx]
# df_train=df_all
# define dataset

class PrivacyDataset(Dataset):
    def __init__(self, df):
        self.sentences = df['review'].values
        self.labels = df["label"].values
        self.embeddingmodel = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(self.labels[idx],idx)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.float32)
        text = self.sentences[idx]
        # print(text)
        a = self.tokenizer.encode(text, add_special_tokens=True)
        embedding_res = self.embeddingmodel(torch.tensor(a).unsqueeze(0))[1].squeeze().detach()

        return embedding_res,label


train_dataset = PrivacyDataset(df_train)
valid_dataset = PrivacyDataset(df_valid)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True,  num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=8)
CFG.steps_per_epoch = len(train_loader)

# model
class CustomEffNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(CFG.embed_dim, CFG.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(CFG.hidden_dim, CFG.num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return x


class LitPrivacy(pl.LightningModule):
    def __init__(self, model):
        super(LitPrivacy, self).__init__()
        self.model = model
        self.metric = torchmetrics.Accuracy(threshold=0.5, num_classes=CFG.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = CFG.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             epochs=CFG.num_epochs, steps_per_epoch=CFG.steps_per_epoch,
                                                             max_lr=CFG.max_lr, pct_start=CFG.pct_start,
                                                             div_factor=CFG.div_factor,
                                                             final_div_factor=CFG.final_div_factor)
        scheduler = {'scheduler': self.scheduler, 'interval': 'step', }

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        embedding_res,label = batch
        label = label.long()
        output = self.model(embedding_res)
        loss = self.criterion(output, label)
        score = self.metric(output.argmax(1).cpu(), label.cpu())
        logs = {'train_loss': loss, 'train_acc': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        embedding_res,label = batch
        label = label.long()
        output = self.model(embedding_res)
        loss = self.criterion(output, label)
        score = self.metric(output.argmax(1).cpu(), label.cpu())
        logs = {'valid_loss': loss, 'valid_acc': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss



if __name__ == "__main__":
    model = CustomEffNet()
    lit_model = LitPrivacy(model.model)
    logger = CSVLogger(save_dir='logs/', name=CFG.model_name)
    logger.log_hyperparams(CFG.__dict__)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                        save_top_k=5,
                                        save_last=True,
                                        save_weights_only=True,
                                        filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
                                        verbose=False,
                                        mode='min')

    trainer = Trainer(
        max_epochs=CFG.num_epochs,
        gpus=[0],
        accumulate_grad_batches=CFG.accum,
        callbacks=[checkpoint_callback],
        logger=logger,
        weights_summary='top',
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    