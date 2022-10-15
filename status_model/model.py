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
from utils import get_avg_metrics

def masked_logits_loss(logits, labels, masks=None, loss_weighting="mean", pos_weight=None, loss_type='bce', focal_gamma=2.):
    # treat unlabeled samples(-1) as implict negative (0.)
    labels2 = torch.clamp_min(labels, 0.)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none', pos_weight=pos_weight)
    if loss_type == 'focal':
        # ref: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/14
        pt = torch.exp(-losses) # prevents nans when probability 0
        losses = (1-pt) ** focal_gamma * losses
    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        if loss_weighting == 'mean':
            return masked_losses.mean()
        elif loss_weighting == 'geo_mean':
            # background ref: https://kexue.fm/archives/8870
            # numerical implementation: https://stackoverflow.com/questions/59722983/how-to-calculate-geometric-mean-in-a-differentiable-way
            masked_losses = masked_losses.mean(0)  # loss per task
            return torch.exp(torch.mean(torch.log(masked_losses)))
    else:
        if loss_weighting == 'mean':
            return losses.mean()
        elif loss_weighting == 'geo_mean':
            losses = losses.mean(0)
            return torch.exp(torch.mean(torch.log(losses)))

def masked_bce_prob_loss(logits, targets, masks=None):
    """
    For continous probability targets
    """
    probs = torch.sigmoid(logits)
    targets = torch.clamp_min(targets, 0.)
    eps = 1e-8
    losses = targets * torch.log(probs + eps) + (1-targets) * torch.log(1 - probs + eps)
    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        return -masked_losses.mean()
    else:
        return -losses.mean()

class LightningInterface(pl.LightningModule):
    def __init__(self, symps, threshold=0.5, loss_mask=True, loss_weighting='mean', loss_type='bce', uncertain='exclude', **kwargs):
        super().__init__()
        self.best_auc = 0.
        self.best_mae = 1.
        self.threshold = threshold
        self.n_classes = len(symps)
        self.loss_mask = loss_mask
        self.loss_weighting = loss_weighting
        self.pos_weight = None
        self.loss_type = loss_type
        self.uncertain = uncertain

    def loss_fn(self, logits, labels, masks):
        return masked_logits_loss(logits, labels, masks, self.loss_weighting, self.pos_weight, self.loss_type)

    def count_loss(self, y_hat, y, masks=None):
        if self.uncertain == "exclude":
            loss = self.loss_fn(y_hat, y, masks)
        elif self.uncertain == "include":
            loss1 = self.loss_fn(y_hat[:, :-1], y[:, :-1], masks[:, :-1])
            loss2 = masked_bce_prob_loss(y_hat[:, -1:], y[:, -1:], masks[:, -1:])
            loss = loss1 + loss2
        elif self.uncertain == "only":
            loss = masked_bce_prob_loss(y_hat[:, -1:], y[:, -1:], masks[:, -1:])
        return loss
    
    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y, masks = batch
        y_hat = self(x)
        if not self.loss_mask:
            masks = None
        loss = self.count_loss(y_hat, y, masks)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        x, y, masks = batch
        y_hat = self(x)
        if not self.loss_mask:
            masks = None
        loss = self.count_loss(y_hat, y, masks)
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'val_loss': loss, "labels": yy, "probs": yy_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().detach().cpu().item()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        tensorboard_logs = {'val_loss': avg_loss}
        if self.uncertain == "exclude":
            ret = get_avg_metrics(all_labels, all_probs, self.threshold)
            self.best_auc = max(self.best_auc, ret['macro_auc'])
            tensorboard_logs['hp_metric'] = self.best_auc
            for k, v in ret.items():
                tensorboard_logs[f"val_{k}"] = v
            self.log("best_auc", self.best_auc, prog_bar=True, on_epoch=True)
        elif self.uncertain == "include":
            ret = get_avg_metrics(all_labels[:, :-1], all_probs[:, :-1], self.threshold)
            self.best_auc = max(self.best_auc, ret['macro_auc'])
            tensorboard_logs['hp_metric'] = self.best_auc
            for k, v in ret.items():
                tensorboard_logs[f"val_{k}"] = v
            self.log("best_auc", self.best_auc, prog_bar=True, on_epoch=True)

            uncertain_probs = all_probs[:, -1]
            uncertain_targets = all_labels[:, -1]
            mae = np.abs(uncertain_probs - uncertain_targets).mean()
            tensorboard_logs["val_uncertain_mae"] = mae
        elif self.uncertain == "only":
            uncertain_probs = all_probs[:, -1]
            uncertain_targets = all_labels[:, -1]
            mae = np.abs(uncertain_probs - uncertain_targets).mean()
            tensorboard_logs["val_uncertain_mae"] = mae
            self.best_mae = min(self.best_mae, mae)
            tensorboard_logs['hp_metric'] = self.best_mae
            self.log("best_mae", self.best_mae, prog_bar=True, on_epoch=True)
        self.log_dict(tensorboard_logs)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y, masks = batch
        y_hat = self(x)
        if not self.loss_mask:
            masks = None
        loss = self.count_loss(y_hat, y, masks)
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'test_loss': loss, "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().detach().cpu().item()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        results = {'test_loss': avg_loss}

        if self.uncertain == "exclude":
            ret = get_avg_metrics(all_labels, all_probs, self.threshold)
            for k, v in ret.items():
                results[f"test_{k}"] = v
        elif self.uncertain == "include":
            ret = get_avg_metrics(all_labels[:, :-1], all_probs[:, :-1], self.threshold)
            for k, v in ret.items():
                results[f"test_{k}"] = v
            uncertain_probs = all_probs[:, -1]
            uncertain_targets = all_labels[:, -1]
            mae = np.abs(uncertain_probs - uncertain_targets).mean()
            results["test_uncertain_mae"] = mae
        elif self.uncertain == "only":
            uncertain_probs = all_probs[:, -1]
            uncertain_targets = all_labels[:, -1]
            mae = np.abs(uncertain_probs - uncertain_targets).mean()
            results["test_uncertain_mae"] = mae

        self.log_dict(results)
        return results

    def on_after_backward(self):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--loss_mask", action='store_true')
        parser.add_argument("--loss_weighting", default="mean", choices=['mean', 'geo_mean'])
        return parser


class Classifier(LightningInterface):
    def __init__(self, symps, threshold=0.5, lr=5e-5, model_type="prajjwal1/bert-tiny", uncertain="exclude", **kwargs):
        super().__init__(symps=symps, threshold=threshold, uncertain=uncertain, **kwargs)

        self.model_type = model_type
        self.model = BERTDiseaseClassifier(model_type, num_symps=len(symps))
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters(ignore=['symps'])
        print(self.hparams)

    def forward(self, x):
        logits = self.model(**x)
        return logits

    def feat_extract(self, x):
        x, logits = self.model.feat_extract(**x)
        return x, logits

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--lr", type=float, default=2e-4)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # if self.lr_sched == "none":
        #     return optimizer
        # elif self.lr_sched == 'reduce':
        #     return {
        #             'optimizer': optimizer, 
        #             'lr_scheduler': {
        #                 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2),
        #                 'monitor': 'val_acc'
        #             }
        #         }
        # else:
        #     schedulers = {
        #         'exp': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8),
        #         'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3),
        #         'cyclic': optim.lr_scheduler.CyclicLR(optimizer, 5e-5, 3e-4, 2, cycle_momentum=False)
        #     }
        #     return {'optimizer': optimizer, 'lr_scheduler': schedulers[self.lr_sched]}


class BERTDiseaseClassifier(nn.Module):
    def __init__(self, model_type, num_symps) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_symps = num_symps
        # multi-label binary classification
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_symps)
    
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]   # [CLS] pooling
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits

    def feat_extract(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]   # [CLS] pooling
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(x)
        logits = self.clf(x)
        return x, logits
