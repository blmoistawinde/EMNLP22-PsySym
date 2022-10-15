
import os
import re
import json
import gzip
import xopen  # faster gzip read
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

def tokenize_text(x):
    return " ".join(word_tokenize(x))

# single TF-IDF vectorizer trained on multi-disease union
# multiple LRs for each single disease against shared control

out_dir = "./baseline_outputs_notok/"
os.makedirs(out_dir, exist_ok=True)

max_vocab = 50000
patient_dir = "../data/patient_user_posts.jl.gz"
control_dir = "../data/control_user_posts.jl.gz"
split_dir = "aid2split.json"
with open(split_dir, "r", encoding="utf-8") as f:
    aid2split = json.load(f)
split_disease2ids = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
split_control2ids = {"train": [], "val": [], "test": []}
split2texts = {"train": [[] for i in range(100000)], "val": [[] for i in range(100000)], "test": [[] for i in range(100000)]}
split_cnts = {"train": 0, "val": 0, "test": 0}

with xopen.xopen(patient_dir) as f:
    for i, line in enumerate(tqdm(f, total=5624)):
        record = json.loads(line)
        diseases = record["diseases"]
        aid = "P" + str(record['id'])
        split = aid2split[aid]
        id_in_split = split_cnts[split]
        text0 = " ".join(record["posts"])
        for disease in diseases:
            split_disease2ids[split][disease].append(id_in_split)
        split2texts[split][id_in_split] = text0
        split_cnts[split] += 1

with xopen.xopen(control_dir) as f:
    for i, line in enumerate(tqdm(f, total=20981)):
        record = json.loads(line)
        aid = "C" + str(record['id'])
        split = aid2split[aid]
        id_in_split = split_cnts[split]
        text0 = " ".join(record["posts"])
        split_control2ids[split].append(id_in_split)
        split2texts[split][id_in_split] = text0
        split_cnts[split] += 1

for split in split2texts:
    split2texts[split] = split2texts[split][:split_cnts[split]]

print("TF-IDF")
tfidf_model = TfidfVectorizer(min_df=5, max_features=max_vocab, stop_words="english")
X_train = tfidf_model.fit_transform(iter(split2texts['train']))
print(X_train.shape)

with open(out_dir+"tfidf.pkl", "wb") as f:
    pickle.dump(tfidf_model, f)

X_val = tfidf_model.transform(iter(split2texts['val']))
print(X_val.shape)

X_test = tfidf_model.transform(iter(split2texts['test']))
print(X_test.shape)

with open(out_dir+"tfidf_vecs.pkl", "wb") as f:
    pickle.dump([X_train, X_val, X_test], f)

results_df = []

# train one disease at a time
for disease in split_disease2ids['train'].keys():
    disease_train_ids = split_disease2ids['train'][disease]
    disease_X_train = X_train[disease_train_ids+split_control2ids['train']]
    disease_test_ids = split_disease2ids['test'][disease]
    disease_X_test = X_test[disease_test_ids+split_control2ids['test']]
    Y_train = np.array([1]*len(disease_train_ids)+[0]*len(split_control2ids['train']))
    Y_test = np.array([1]*len(disease_test_ids)+[0]*len(split_control2ids['test']))

    lr = LogisticRegression()
    lr.fit(disease_X_train, Y_train)
    Y_pred_probs = lr.predict_proba(disease_X_test)
    Y_preds = (Y_pred_probs[:, 1] > 0.5).astype(int)
    acc = accuracy_score(Y_test, Y_preds)
    p = precision_score(Y_test, Y_preds)
    r = recall_score(Y_test, Y_preds)
    f1 = f1_score(Y_test, Y_preds)
    auc = roc_auc_score(Y_test, Y_pred_probs[:, 1])
    print(disease)
    print(f"Acc: {acc:.4f}, P: {p:.4f}, R: {r:.4f}, F: {f1:.4f}, AUC: {auc:.4f}")
    results_df.append([disease, p, r, f1, auc])
    with open(out_dir + f"lr_{disease}.pkl", "wb") as f:
        pickle.dump(lr, f)

results_df = pd.DataFrame(results_df)
results_df.columns = ['disease', 'p', 'r', 'f1', 'auc']
results_df = results_df.set_index("disease")
print(results_df)
results_df.to_csv(out_dir+"LR_exp_results.csv", encoding="utf-8")

"""
max_vocab = 50000
                      p         r        f1       auc
disease
depression     0.914141  0.553517  0.689524  0.922837
anxiety        0.918519  0.521008  0.664879  0.940141
autism         1.000000  0.178082  0.302326  0.872264
adhd           0.929825  0.438017  0.595506  0.908079
schizophrenia  1.000000  0.205128  0.340426  0.958255
bipolar        0.975309  0.506410  0.666667  0.954027
ocd            1.000000  0.150000  0.260870  0.884342
ptsd           1.000000  0.179487  0.304348  0.920131
eating         1.000000  0.058824  0.111111  0.973051

max_vocab = 100000 gets similar results
 
"""
