import os
import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from utils import PosWeightCallback
from data import MultiDiseaseDataModule
from model import Classifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, truncation=True)
    data_module = MultiDiseaseDataModule(args.bs, args.input_dir, tokenizer, args.max_len, args.uncertain, args.bal_sample, args.control_ratio)
    model = model_type(symps=data_module.symps, **vars(args))
    if args.uncertain != "only":
        early_stop_callback = EarlyStopping(
            monitor='val_macro_auc',
            patience=args.patience,
            mode="max"
        )
        checkpoint_callback = ModelCheckpoint(monitor='val_macro_auc', save_top_k=1, save_weights_only=True, mode='max')
    else:
        early_stop_callback = EarlyStopping(
            monitor='val_uncertain_mae',
            patience=args.patience,
            mode="min"
        )
        checkpoint_callback = ModelCheckpoint(monitor='val_uncertain_mae', save_top_k=1, save_weights_only=True, mode='min')
    
    pos_callback = PosWeightCallback(args.pos_weight_setting)
    
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback, pos_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=10)

    trainer.fit(model, data_module)
    results = trainer.test(datamodule=data_module)
    result = results[0]
    if args.write_result_dir != "" and args.exp_name != "":
        result['exp_name'] = args.exp_name
        result['seed'] = args.seed
        with open(args.write_result_dir, "a", encoding="utf-8") as f:
            f.write(json.dumps(result)+"\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="prajjwal1/bert-tiny")
    # parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    temp_args, _ = parser.parse_known_args()
    model_type = Classifier
    parser = model_type.add_model_specific_args(parser)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--input_dir", type=str, default="../data/symp_data")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--uncertain", type=str, default='exclude', choices=['include', 'exclude', 'only'])
    parser.add_argument("--pos_weight_setting", type=str, default='default')
    parser.add_argument("--loss_type", type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument("--bal_sample", action='store_true')
    parser.add_argument("--control_ratio", type=float, default=0.75)
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--write_result_dir", default="")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    main(args)