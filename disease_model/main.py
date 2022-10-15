import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from data import SingleDiseaseDataModule
from model import Classifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def main(args):
    data_module = SingleDiseaseDataModule(args.bs, args.input_dir, args.max_len, args.feat_type, args.use_subj, args.use_uncertain, args.sel_disease, args.concat_feats)
    data_module.prepare_data()
    data_module.setup("fit")
    model = model_type(in_dim=len(data_module.train_set.data[0][0]), **vars(args))
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=args.patience,
        mode="max"
    )
    checkpoint_callback = ModelCheckpoint(monitor='val_f1', save_top_k=1, save_weights_only=True, mode='max')
    
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=10)
    trainer.fit(model, data_module)
    results = trainer.test(datamodule=data_module)
    result = results[0]
    if args.write_result_dir != "" and args.exp_name != "":
        result["sel_disease"] = args.sel_disease
        result["use_subj"] = args.use_subj
        result["use_uncertain"] = args.use_uncertain
        result["feat_type"] = args.feat_type
        result["concat_feats"] = args.concat_feats
        result['exp_name'] = args.exp_name
        result['seed'] = args.seed
        with open(args.write_result_dir, "a", encoding="utf-8") as f:
            f.write(json.dumps(result)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    temp_args, _ = parser.parse_known_args()
    model_type = Classifier
    parser = model_type.add_model_specific_args(parser)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--input_dir", type=str, default="./symp_dataset_tiny")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--feat_type", default="prob", choices=['prob', 'emb'])
    parser.add_argument("--sel_disease", default="depression", choices=['depression', 'anxiety', 'autism', 'adhd', 'schizophrenia', 'bipolar', 'ocd', 'ptsd', 'eating'])
    parser.add_argument("--use_subj", action="store_true")
    parser.add_argument("--use_uncertain", action="store_true")
    parser.add_argument("--concat_feats", action="store_true")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--write_result_dir", default="")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    main(args)