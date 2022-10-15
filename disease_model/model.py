import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report

class LightningInterface(pl.LightningModule):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y, masks = batch
        y_hat = self(x, masks)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        x, y, masks = batch
        y_hat = self(x, masks)
        loss = self.criterion(y_hat, y)
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'val_loss': loss, "labels": yy, "probs": yy_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().detach().cpu().item()
        all_labels = np.concatenate([x['labels'] for x in outputs])[:, 0]
        all_probs = np.concatenate([x['probs'] for x in outputs])[:, 0]
        all_preds = (all_probs > self.threshold).astype(float)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        f1 = f1_score(all_labels, all_preds)
        self.best_f1 = max(self.best_f1,f1)
        tensorboard_logs = {'val_loss': avg_loss, 'hp_metric': self.best_f1}
        tensorboard_logs[f"val_f1"] = f1
        tensorboard_logs[f"val_auc"] = auc
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y, masks = batch
        y_hat = self(x, masks)
        loss = self.criterion(y_hat, y)
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'test_loss': loss, "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().detach().cpu().item()
        all_labels = np.concatenate([x['labels'] for x in outputs])[:, 0]
        all_probs = np.concatenate([x['probs'] for x in outputs])[:, 0]
        all_preds = (all_probs > self.threshold).astype(float)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        f1 = f1_score(all_labels, all_preds)
        results = {'test_loss': avg_loss}
        results["test_f1"] = f1
        results["test_auc"] = auc
        self.log_dict(results)
        return results

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--threshold", type=float, default=0.5)
        return parser


class Classifier(LightningInterface):
    def __init__(self, in_dim, threshold=0.5, lr=0.01, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model = KMaxMeanCNN(in_dim)
        self.lr = lr
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, seq, seq_masks):
        logits = self.model(seq, seq_masks)
        return logits

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--lr", type=float, default=0.01)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# https://discuss.pytorch.org/t/resolved-how-to-implement-k-max-pooling-for-cnn-text-classification/931
# def kmax_pooling(x, dim, k):
#     index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
#     return x.gather(dim, index)

# no relative position, but can be deterministic
def kmax_pooling(x, k):
    return x.sort(dim = 2)[0][:, :, -k:]

class KMaxMeanCNN(nn.Module):
    def __init__(self, in_dim, filter_num=50, filter_sizes=(2,3,4,5,6), dropout=0.2, max_pooling_k=5):
        super(KMaxMeanCNN, self).__init__()
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.max_pooling_k = max_pooling_k
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_dim, filter_num, size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, input_seqs, seq_masks=None):
        # import ipdb; ipdb.set_trace()
        input_seqs = input_seqs.transpose(1, 2) # [bs, L, in_dim] -> [bs, in_dim, L]
        x = [F.relu(conv(input_seqs)) for conv in self.convs]
        x = [kmax_pooling(item, self.max_pooling_k).mean(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
