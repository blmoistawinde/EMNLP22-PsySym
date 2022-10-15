import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from model import Classifier, BERTDiseaseClassifier
from utils import default_symps

if __name__ == "__main__":
    batch_size = 64
    patient_dir = "../data/patient_user_posts.jl.gz"
    control_dir = "../data/control_user_posts.jl.gz"
    split_dir = "../data/aid2split.json"
    with open(split_dir, "r", encoding="utf-8") as f:
        aid2split = json.load(f)
    output_dir = "../disease_model/dataset_org_bert_avg/"
    os.makedirs(output_dir, exist_ok=True)
    model_type = "bert-base-uncased"
    max_len = 64
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # model weight directly load from pretrained bert
    # symp probs will be random
    clf = BERTDiseaseClassifier(model_type, len(default_symps))
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
            for post in record["posts"]:
                sents = cut_sentences(post)
                curr_sid += len(sents)
                sent_bounds.append(curr_sid)
                user_sents.extend(sents)

            all_probs = []
            all_feats = []
            for i in range(0, len(user_sents), batch_size):
                curr_texts = user_sents[i:i+batch_size]
                processed_batch = infer_preprocess(curr_texts, tokenizer, max_len)
                for k, v in processed_batch.items():
                    processed_batch[k] = v.cuda()
                with torch.no_grad():
                    feats, logits = clf.feat_extract_avg(**processed_batch)
                    feats = feats.detach().cpu().numpy()
                    probs = logits.sigmoid().detach().cpu().numpy()
                all_probs.append(probs)
                all_feats.append(feats)
            all_probs = np.concatenate(all_probs, 0)
            all_feats = np.concatenate(all_feats, 0)

            # merge all sentence features into post-level feature by max pooling
            all_post_probs = []
            all_post_feats = []
            for i in range(len(sent_bounds)-1):
                lbound, rbound = sent_bounds[i], sent_bounds[i+1]
                post_prob = all_probs[lbound:rbound, :].max(0)
                all_post_probs.append(post_prob)
                post_feat = all_feats[lbound:rbound, :].mean(0)
                all_post_feats.append(post_feat)
            all_post_probs = np.stack(all_post_probs, 0)
            all_post_feats = np.stack(all_post_feats, 0)
            data = {
                "id": aid,
                "diseases": record["diseases"],
                "probs": all_post_probs,
                "feats": all_post_feats
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
            for post in record["posts"]:
                sents = cut_sentences(post)
                curr_sid += len(sents)
                sent_bounds.append(curr_sid)
                user_sents.extend(sents)
            
            all_probs = []
            all_feats = []
            for i in range(0, len(user_sents), batch_size):
                curr_texts = user_sents[i:i+batch_size]
                processed_batch = infer_preprocess(curr_texts, tokenizer, max_len)
                for k, v in processed_batch.items():
                    processed_batch[k] = v.cuda()
                with torch.no_grad():
                    feats, logits = clf.feat_extract_avg(**processed_batch)
                    feats = feats.detach().cpu().numpy()
                    probs = logits.sigmoid().detach().cpu().numpy()
                all_probs.append(probs)
                all_feats.append(feats)
            all_probs = np.concatenate(all_probs, 0)
            all_feats = np.concatenate(all_feats, 0)

            # merge all sentence features into post-level feature by max pooling
            all_post_probs = []
            all_post_feats = []
            for i in range(len(sent_bounds)-1):
                lbound, rbound = sent_bounds[i], sent_bounds[i+1]
                post_prob = all_probs[lbound:rbound, :].max(0)
                all_post_probs.append(post_prob)
                post_feat = all_feats[lbound:rbound, :].mean(0)
                all_post_feats.append(post_feat)
            all_post_probs = np.stack(all_post_probs, 0)
            all_post_feats = np.stack(all_post_feats, 0)
            data = {
                "id": aid,
                "diseases": record["diseases"],
                "probs": all_post_probs,
                "feats": all_post_feats
            }
            split2dataset[split].append(data)

print("Writing")
for split, dataset in split2dataset.items():
    with open(os.path.join(output_dir, f"{split}.pkl"), "wb") as fo:   
        pickle.dump(dataset, fo)
