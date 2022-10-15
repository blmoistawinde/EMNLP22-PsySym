import os
import re
import torch
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

class MultiDiseaseDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, uncertain, split="train"):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.uncertain = uncertain
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, f"{split}.csv")
        df = pd.read_csv(input_dir2, index_col=None)
        if uncertain == 'exclude':
            self.symps = df.columns[5:].to_list()
            self.symps.remove("uncertain")
        elif uncertain == 'include':
            self.symps = df.columns[5:].to_list()
        else:
            self.symps = ['uncertain']

        self.is_control = []

        for rid, row in df.iterrows():
            sample = {}
            sample["text"] = row['sentence']
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            if uncertain == 'exclude':
                self.labels.append(row.values[5:-1])
            elif uncertain == 'include':
                self.labels.append(row.values[5:])
            else:
                self.labels.append(row.values[-1:])
            self.is_control.append(row['disease'] == 'control')

        self.is_control = np.array(self.is_control).astype(int)
        self.label_counters = torch.zeros(len(self.symps), 2)
        for labels0 in self.labels:
            for class_id, label in enumerate(labels0):
                label = int(label)
                if label in {0, 1}:
                    self.label_counters[class_id, label] += 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_disease(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        if k in processed_batch:  # roberta has no token_type_ids
            processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    masks = torch.not_equal(labels, -1)
    return processed_batch, labels, masks

def preprocess_sent(sent):
    # remove hyperlink and preserve text mention
    sent = re.sub('\[(.*?)\]\(.*?\)', r'\1', sent)
    sent = sent.replace("[removed]", "")
    sent = sent.strip()
    return sent

def infer_preprocess(texts, tokenizer, max_len):
    texts = [preprocess_sent(text) for text in texts]
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    processed_batch = {}
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        if k in tokenized:  # roberta has no token_type_ids
            processed_batch[k] = torch.LongTensor(tokenized[k])
    return processed_batch

class BalanceSampler(Sampler):
    def __init__(self, data_source, control_ratio=0.75) -> None:
        self.data_source = data_source
        self.control_ratio = control_ratio
        self.indexes_control = np.where(data_source.is_control == 1)[0]
        self.indexes_mental = np.where(data_source.is_control == 0)[0]
        self.len_control = len(self.indexes_control)
        self.len_mental = len(self.indexes_mental)

        np.random.shuffle(self.indexes_control)
        np.random.shuffle(self.indexes_mental)

        self.pointer_control = 0
        self.pointer_mental = 0

    def __iter__(self):
        for i in range(len(self.data_source)):
            if np.random.rand() < self.control_ratio:
                id0 = np.random.randint(self.pointer_control, self.len_control)
                sel_id = self.indexes_control[id0]
                self.indexes_control[id0], self.indexes_control[self.pointer_control] = self.indexes_control[self.pointer_control], self.indexes_control[id0]
                self.pointer_control += 1
                if self.pointer_control >= self.len_control:
                    self.pointer_control = 0
                    np.random.shuffle(self.indexes_control)
            else:
                id0 = np.random.randint(self.pointer_mental, self.len_mental)
                sel_id = self.indexes_mental[id0]
                self.indexes_mental[id0], self.indexes_mental[self.pointer_mental] = self.indexes_mental[self.pointer_mental], self.indexes_mental[id0]
                self.pointer_mental += 1
                if self.pointer_mental >= self.len_mental:
                    self.pointer_mental = 0
                    np.random.shuffle(self.indexes_mental)
            
            yield sel_id

    def __len__(self) -> int:
        return len(self.data_source)


class MultiDiseaseDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len, uncertain, bal_sample=False, control_ratio=0.75):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.uncertain = uncertain
        self.bal_sample = bal_sample
        self.control_ratio = control_ratio
        if uncertain == 'only':
            self.symps = ['uncertain']
        else:
            self.symps = pd.read_csv(os.path.join(input_dir, f"test.csv"), index_col=None).columns[5:].to_list()
            if uncertain == 'exclude':
                self.symps.remove("uncertain")
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = MultiDiseaseDataset(self.input_dir, self.tokenizer, self.max_len, self.uncertain, "train")
            self.val_set = MultiDiseaseDataset(self.input_dir, self.tokenizer, self.max_len, self.uncertain, "val")
            self.test_set = MultiDiseaseDataset(self.input_dir, self.tokenizer, self.max_len, self.uncertain, "test")
        elif stage == "test":
            self.test_set = MultiDiseaseDataset(self.input_dir, self.tokenizer, self.max_len, self.uncertain, "test")
        

    def train_dataloader(self):
        if self.bal_sample:
            sampler = BalanceSampler(self.train_set, self.control_ratio)
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_disease, sampler=sampler, pin_memory=True, num_workers=4)
        else:
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_disease, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_disease, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_disease, pin_memory=True, num_workers=4)

if __name__ == "__main__":
    model_id = 'prajjwal1/bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_module = MultiDiseaseDataModule(2, "../data/symp_data", tokenizer, max_len=64)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    batch0, labels0 = next(iter(train_loader))
    print(batch0)
    print(labels0)
    import ipdb; ipdb.set_trace()