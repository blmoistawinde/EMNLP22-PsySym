import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict
import blingfire
cut_sentences = lambda x: blingfire.text_to_sentences(x.strip()).split("\n")

class SingleDiseaseDataset(Dataset):
    def __init__(self, input_dir, max_len, feat_type, use_subj, use_uncertain, sel_disease, concat_feats=False, split="train"):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.max_len = max_len
        self.feat_type = feat_type
        self.use_subj = use_subj
        self.use_uncertain = use_uncertain
        self.sel_disease = sel_disease
        self.concat_feats = concat_feats
        self.data = []
        self.seq_masks = []
        self.labels = []
        main_feats = pickle.load(open(os.path.join(input_dir, f"{split}.pkl"), "rb"))
        other_feats = pickle.load(open(os.path.join(input_dir, f"{split}_other_feats.pkl"), "rb"))
        for main_feat, other_feat in zip(main_feats, other_feats):
            diseases = main_feat["diseases"]
            if len(diseases) > 0 and sel_disease not in diseases:
                continue
            probs = main_feat['probs']
            if use_uncertain:
                probs = probs * (1 - other_feat['uncertain']).reshape(-1, 1)
            if use_subj:
                probs = probs * (0.1 + 0.8 * other_feat['subj']).reshape(-1, 1)
            embs = main_feat["feats"]
            
            if feat_type == "prob":
                feat = probs
            elif feat_type == "emb":
                feat = embs

            if concat_feats:
                feat = np.concatenate([feat, (1 - other_feat['uncertain']).reshape(-1, 1), other_feat['subj'].reshape(-1, 1)], axis=1)
            if feat.shape[0] < max_len:
                # zero padding at the right
                seq_mask = np.concatenate([np.ones((feat.shape[0],)), np.zeros((max_len-feat.shape[0],))])
                feat = np.concatenate([feat, np.zeros((max_len-feat.shape[0], feat.shape[1]))])
            elif feat.shape[0] >= max_len:
                seq_mask = np.ones((max_len, ))
                feat = feat[:max_len, :]
            self.data.append(feat)
            self.seq_masks.append(seq_mask)
            self.labels.append(np.array([int(len(diseases) > 0)]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index], self.seq_masks[index]


def preprocess_sent(sent):
    # remove hyperlink and preserve text mention
    sent = re.sub('\[(.*?)\]\(.*?\)', r'\1', sent)
    sent = sent.replace("[removed]", "")
    sent = sent.strip()
    return sent


def my_collate_fn(batch):
    data = []
    labels = []
    masks = []
    for d, l, m in batch:
        data.append(d)
        labels.append(l)
        masks.append(m)
    data = torch.FloatTensor(data)
    labels = torch.FloatTensor(labels)
    masks = torch.FloatTensor(masks)
    return data, labels, masks

class SingleDiseaseDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, max_len, feat_type, use_subj, use_uncertain, sel_disease, concat_feats):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.max_len = max_len
        self.feat_type = feat_type
        self.use_subj = use_subj
        self.use_uncertain = use_uncertain
        self.sel_disease = sel_disease
        self.concat_feats = concat_feats
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = SingleDiseaseDataset(self.input_dir, self.max_len, self.feat_type, self.use_subj, self.use_uncertain, self.sel_disease, self.concat_feats, "train")
            self.val_set = SingleDiseaseDataset(self.input_dir, self.max_len, self.feat_type, self.use_subj, self.use_uncertain, self.sel_disease, self.concat_feats, "val")
            self.test_set = SingleDiseaseDataset(self.input_dir, self.max_len, self.feat_type, self.use_subj, self.use_uncertain, self.sel_disease, self.concat_feats, "test")
        elif stage == "test":
            self.test_set = SingleDiseaseDataset(self.input_dir, self.max_len, self.feat_type, self.use_subj, self.use_uncertain, self.sel_disease, self.concat_feats, "test")
        

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=4, collate_fn=my_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, pin_memory=True, num_workers=4, collate_fn=my_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, pin_memory=True, num_workers=4, collate_fn=my_collate_fn)

if __name__ == "__main__":
    data_module = SingleDiseaseDataModule(4, "./symp_dataset_tiny", 256, 'emb', True, True, 'depression')
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    batch0, labels0, masks0 = next(iter(test_loader))
    print(batch0.shape)
    print(masks0)
    print(labels0)
    import ipdb; ipdb.set_trace()