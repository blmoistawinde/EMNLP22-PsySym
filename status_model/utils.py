import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from nltk import word_tokenize
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer

self_mentions = set(["i", "me", "myself"])

ref_mentions = [
        "aunt",
        "aunty",
        "baby",
        "boy",
        "brother",
        "child",
        "cousin",
        "dad",
        "daughter",
        "father",
        "friend",
        "friends",
        "girl",
        "grandfather",
        "grandmother",
        "he",
        "he's",
        "her",
        "hers",
        "him",
        "his",
        "husband",
        "mom",
        "mother",
        "people",
        "persons",
        "she",
        "she's",
        "sister",
        "son",
        "their",
        "they",
        "uncle",
        "wife",
        "himself",
        "herself",
        "themselves"
    ]
ref_mentions = set(ref_mentions)

pattern_self = re.compile(rf"\b({'|'.join(self_mentions)})\b", re.IGNORECASE)
pattern_ref = re.compile(rf"\b({'|'.join(ref_mentions)})\b", re.IGNORECASE)

def decide_subject(text):
    # if the subject is the speaker, then return 1
    self_cnt = len(pattern_self.findall(text))
    ref_cnt = len(pattern_ref.findall(text))
    return int(self_cnt >= ref_cnt)

class PosWeightCallback(Callback):
    def __init__(self, pos_weight_setting='default') -> None:
        super().__init__()
        self.pos_weight_setting = pos_weight_setting
        
    def on_fit_start(self, trainer, pl_module):
        if self.pos_weight_setting == 'default':
            return
        elif self.pos_weight_setting == 'balance':
            # get pos weight stats from train dataset
            label_counters = trainer.datamodule.train_set.label_counters
            pl_module.pos_weight = label_counters[: ,0] / label_counters[: ,1]
            pl_module.pos_weight = pl_module.pos_weight.to(pl_module.device)
        elif self.pos_weight_setting == 'sqrt_balance':
            label_counters = trainer.datamodule.train_set.label_counters
            pl_module.pos_weight = torch.sqrt(label_counters[: ,0] / label_counters[: ,1])
            pl_module.pos_weight = pl_module.pos_weight.to(pl_module.device)
        else:
            # a constant float for all classes
            pl_module.pos_weight = torch.FloatTensor([float(self.pos_weight_setting)] * pl_module.n_classes)
            pl_module.pos_weight = pl_module.pos_weight.to(pl_module.device)


def get_avg_metrics(all_labels, all_probs, threshold):
    labels_by_class = []
    probs_by_class = []
    for i in range(all_labels.shape[1]):
        sel_indices = np.where(all_labels[:, i] != -1)
        labels_by_class.append(all_labels[:, i][sel_indices])
        probs_by_class.append(all_probs[:, i][sel_indices])
    # macro avg metrics
    ret = {}
    for k in ["macro_acc", "macro_p", "macro_r", "macro_f", "macro_auc"]:
        ret[k] = []
    for labels, probs in zip(labels_by_class, probs_by_class):
        preds = (probs > threshold).astype(float)
        ret["macro_acc"].append(np.mean(labels == preds))
        ret["macro_p"].append(precision_score(labels, preds))
        ret["macro_r"].append(recall_score(labels, preds))
        ret["macro_f"].append(f1_score(labels, preds))
        try:
            ret["macro_auc"].append(roc_auc_score(labels, probs))
        except:
            ret["macro_auc"].append(0.5)
    for k in ["macro_acc", "macro_p", "macro_r", "macro_f", "macro_auc"]:
        ret[k] = np.mean(ret[k])

    # micro metrics
    merged_labels = np.concatenate(labels_by_class)
    merged_probs = np.concatenate(probs_by_class)
    merged_preds = (merged_probs > threshold).astype(float)
    ret["micro_acc"] = np.mean(merged_labels == merged_preds)
    ret["micro_p"] = precision_score(merged_labels, merged_preds)
    ret["micro_r"] = recall_score(merged_labels, merged_preds)
    ret["micro_f"] = f1_score(merged_labels, merged_preds)
    try:
        ret["micro_auc"] = roc_auc_score(merged_labels, merged_probs)
    except:
        ret["micro_auc"] = 0.5
    return ret

default_symps = ['Anxious_Mood',
 'Autonomic_symptoms',
 'Cardiovascular_symptoms',
 'Catatonic_behavior',
 'Decreased_energy_tiredness_fatigue',
 'Depressed_Mood',
 'Gastrointestinal_symptoms',
 'Genitourinary_symptoms',
 'Hyperactivity_agitation',
 'Impulsivity',
 'Inattention',
 'Indecisiveness',
 'Respiratory_symptoms',
 'Suicidal_ideas',
 'Worthlessness_and_guilty',
 'avoidance_of_stimuli',
 'compensatory_behaviors_to_prevent_weight_gain',
 'compulsions',
 'diminished_emotional_expression',
 'do_things_easily_get_painful_consequences',
 'drastical_shift_in_mood_and_energy',
 'fear_about_social_situations',
 'fear_of_gaining_weight',
 'fears_of_being_negatively_evaluated',
 'flight_of_ideas',
 'intrusion_symptoms',
 'loss_of_interest_or_motivation',
 'more_talktive',
 'obsession',
 'panic_fear',
 'pessimism',
 'poor_memory',
 'sleep_disturbance',
 'somatic_muscle',
 'somatic_symptoms_others',
 'somatic_symptoms_sensory',
 'weight_and_appetite_change',
 'Anger_Irritability']
