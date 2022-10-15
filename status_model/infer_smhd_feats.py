import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import yaml
import xopen
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
from os.path import dirname
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from data import infer_preprocess, cut_sentences
from model import Classifier
from utils import decide_subject

if __name__ == "__main__":
    batch_size = 64
    patient_dir = "../data/patient_user_posts.jl.gz"
    control_dir = "../data/control_user_posts.jl.gz"
    split_dir = "../data/aid2split.json"
    with open(split_dir, "r", encoding="utf-8") as f:
        aid2split = json.load(f)
    output_dir = "../disease_model/symp_dataset/"
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = "YOUR TRAINED CKPT"
    hparams_dir = os.path.join(dirname(dirname(ckpt_dir)), 'hparams.yaml')
    hparams = yaml.load(open(hparams_dir))
    max_len = hparams["max_len"]
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_type"])
    clf = Classifier.load_from_checkpoint(ckpt_dir, symps=['uncertain'])
    clf.eval()
    clf.cuda()
    split2dataset = {split: [] for split in ['train', 'val', 'test']}
    
    with xopen.xopen(patient_dir) as fi:
        for i, line in tqdm(enumerate(fi), total=5624):
            record = json.loads(line)
            aid = "P" + str(record['id'])
            split = aid2split[aid]
            user_sents = []
            sent_bounds = [0]
            curr_sid = 0
            post_subj = []
            for post in record["posts"]:
                post_subj.append(decide_subject(post))
                sents = cut_sentences(post)
                curr_sid += len(sents)
                sent_bounds.append(curr_sid)
                user_sents.extend(sents)

            all_probs = []
            for i in range(0, len(user_sents), batch_size):
                curr_texts = user_sents[i:i+batch_size]
                processed_batch = infer_preprocess(curr_texts, tokenizer, max_len)
                for k, v in processed_batch.items():
                    processed_batch[k] = v.to(clf.device)
                with torch.no_grad():
                    logits = clf(processed_batch)
                    probs = logits.sigmoid().detach().cpu().numpy()
                all_probs.append(probs)
            all_probs = np.concatenate(all_probs, 0)

            # merge all sentence features into post-level feature by max pooling
            all_post_probs = []
            for i in range(len(sent_bounds)-1):
                lbound, rbound = sent_bounds[i], sent_bounds[i+1]
                post_prob = all_probs[lbound:rbound, 0].max()
                all_post_probs.append(post_prob)
            all_post_probs = np.array(all_post_probs)
            post_subj = np.array(post_subj)
            data = {
                "id": aid,
                "diseases": record["diseases"],
                "uncertain": all_post_probs,
                "subj": post_subj
            }
            split2dataset[split].append(data)

    with xopen.xopen(control_dir) as fi:
        for i, line in tqdm(enumerate(fi), total=20981):
            record = json.loads(line)
            aid = "C" + str(record['id'])
            split = aid2split[aid]
            user_sents = []
            sent_bounds = [0]
            curr_sid = 0
            post_subj = []
            for post in record["posts"]:
                post_subj.append(decide_subject(post))
                sents = cut_sentences(post)
                curr_sid += len(sents)
                sent_bounds.append(curr_sid)
                user_sents.extend(sents)

            all_probs = []
            for i in range(0, len(user_sents), batch_size):
                curr_texts = user_sents[i:i+batch_size]
                processed_batch = infer_preprocess(curr_texts, tokenizer, max_len)
                for k, v in processed_batch.items():
                    processed_batch[k] = v.to(clf.device)
                with torch.no_grad():
                    logits = clf(processed_batch)
                    probs = logits.sigmoid().detach().cpu().numpy()
                all_probs.append(probs)
            all_probs = np.concatenate(all_probs, 0)

            # merge all sentence features into post-level feature by max pooling
            all_post_probs = []
            for i in range(len(sent_bounds)-1):
                lbound, rbound = sent_bounds[i], sent_bounds[i+1]
                post_prob = all_probs[lbound:rbound, 0].max()
                all_post_probs.append(post_prob)
            all_post_probs = np.array(all_post_probs)
            post_subj = np.array(post_subj)
            data = {
                "id": aid,
                "diseases": record["diseases"],
                "uncertain": all_post_probs,
                "subj": post_subj
            }
            split2dataset[split].append(data)


    print("Writing")
    for split, dataset in split2dataset.items():
        with open(os.path.join(output_dir, f"{split}_other_feats.pkl"), "wb") as fo:   
            pickle.dump(dataset, fo)
