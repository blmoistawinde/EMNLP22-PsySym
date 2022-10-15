import os
import yaml
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from os.path import dirname
import argparse
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from data import MultiDiseaseDataModule, MultiDiseaseDataset, my_collate_disease
from model import Classifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(args):
    ckpt_dir = args.ckpt_dir
    split = args.infer_split
    hparams_dir = os.path.join(dirname(dirname(ckpt_dir)), 'hparams.yaml')
    hparams = yaml.load(open(hparams_dir))
    max_len = hparams["max_len"]
    uncertain = hparams["uncertain"]
    exp_name = hparams["exp_name"]
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_type"])
    dataset = MultiDiseaseDataset(args.infer_input_dir, tokenizer, max_len, uncertain, split)
    dataloader = DataLoader(dataset, batch_size=args.bs, collate_fn=my_collate_disease, pin_memory=True, num_workers=4)
    clf = Classifier.load_from_checkpoint(ckpt_dir, symps=dataset.symps)
    clf.eval()
    clf.cuda()
    all_probs = []
    for batch in tqdm(dataloader, desc="Inference: "):
        x, y, masks = batch
        for k in ['input_ids', 'attention_mask', 'token_type_ids']:
            if k in x:
                x[k] = x[k].cuda()
        with torch.no_grad():
            y_hat = clf(x)
            probs = torch.sigmoid(y_hat)
            # probs = y_hat    # preserve logits here, will do recalibration later
            probs = probs.detach().cpu().numpy()
        del x, y, y_hat, masks
        all_probs.extend(probs)
    all_probs = np.stack(all_probs, 0)
    out_dir = os.path.join(args.infer_output_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    infer_output_file = os.path.join(out_dir, f"{split}.npy")
    np.save(infer_output_file, all_probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--infer_input_dir", type=str, default="../data/symp_data")
    parser.add_argument("--infer_split", type=str, default="test")
    parser.add_argument("--infer_output_dir", type=str, default="./infer_output")
    args = parser.parse_args()
    main(args)