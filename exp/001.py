import os
import gc
import re
import sys
sys.path.append("/root/workspace/Foursquare2022")
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import librosa
import scipy as sp
import numpy as np
import pandas as pd
import soundfile as sf
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm

from contextlib import contextmanager
from pathlib import Path
from typing import List
from typing import Optional
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import transformers


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter


class CFG:
    EXP_ID = '001'
    seed = 71
    epochs = 5
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    train_bs = 16
    valid_bs = 32
    EARLY_STOPPING = True
    DEBUG = False # True
    target = "point_of_interest"
    n_neighbors = 10
    n_splits = 3


import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")



train = pd.read_csv('input/train_with_near_candidate_target.csv')
train['num_target'] = train[[f"target_{i}" for i in range(10)]].sum(1)


train = train.head(10000)


import Levenshtein
import difflib
from requests import get
import multiprocessing
import joblib


def _add_distance_features(args):
    _, df = args

    columns = ['name', 'address', 'city', 'state',
           'zip', 'country', 'url', 'phone', 'categories']

    for i in tqdm(range(CFG.n_neighbors)):
        for c in columns:
            geshs = []
            levens = []
            jaros = []
            lcss = []
            for str1, str2 in df[[f"near_{c}_0", f"near_{c}_{i}"]].values.astype(str):
                if str1==str1 and str2==str2:
                    geshs.append(difflib.SequenceMatcher(None, str1, str2).ratio())
                    levens.append(Levenshtein.distance(str1, str2))
                    jaros.append(Levenshtein.jaro_winkler(str1, str2))
                    #lcss.append(LCS(str(str1), str(str2)))
                else:
                    geshs.append(-1)
                    levens.append(-1)
                    jaros.append(-1)
            df[f"near_{c}_{i}_gesh"] = geshs
            df[f"near_{c}_{i}_leven"] = levens
            df[f"near_{c}_{i}_jaro"] = jaros
            #df[f"near_{c}_{i}_lcs"] = lcss

            if not c in ['country', "phone", "zip"]:
                df[f"near_{c}_{i}_len"] = df[f"near_{c}_{i}"].astype(str).map(len)
                df[f"near_{c}_{i}_nleven"] = df[f"near_{c}_{i}_leven"] / df[[f"near_{c}_{i}_len", f"near_{c}_0_len"]].max(axis=1)
                #df[f"near_{c}_{i}_nlcsi"] = df[f"near_{c}_{i}_lcs"] / df[f"near_{c}_{i}_len"]
                #df[f"near_{c}_{i}_nlcs0"] = df[f"near_{c}_{i}_lcs"] / df[f"near_{c}_0_len"]
    return df


def add_distance_features(df):
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        dfs = pool.imap_unordered(_add_distance_features, df.groupby('country'))
        dfs = tqdm(dfs)
        dfs = list(dfs)
    df = pd.concat(dfs)
    return df

train = add_distance_features(train)


features = []

columns = ['name', 'address', 'city', 'state',
       'zip', 'country', 'url', 'phone', 'categories']
for i in tqdm(range(CFG.n_neighbors)):
    features.append(f"d_near_{i}")
    for c in columns:        
        features += [f"near_{c}_{i}_gesh", f"near_{c}_{i}_jaro"] #, f"near_{c}_{i}_lcs"]
        if c in ['country', "phone", "zip"]:
            features += [f"near_{c}_{i}_leven"]
        else:
            features += [f"near_{c}_{i}_len", f"near_{c}_{i}_nleven"] #, f"near_{c}_{i}_nlcsi", f"near_{c}_{i}_nlcs0"]

print(features)


train = train[features + [CFG.target, "num_target", "id"] + [f"target_{i}" for i in range(10)] + [f"near_id_{i}" for i in range(CFG.n_neighbors)]]
train[features] = train[features].astype(np.float16)
train.reset_index(drop=True, inplace=True)

kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(train, train["num_target"], train["num_target"]))):
    train.loc[val_idx, "fold"] = i


print(train.shape)
# print(train.head())

"""
import lightgbm as lgbm

def fit_lgbm(X, y, params=None, es_rounds=20, seed=42, N_SPLITS=5,
             n_class=None, model_dir=None, folds=None):
    models = []
    oof = np.zeros((len(y), n_class), dtype=np.float64)

    for i in tqdm(range(CFG.n_splits)):
        print(f"== fold {i} ==")
        trn_idx = folds!=i
        val_idx = folds==i
        X_train, y_train = X[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        if model_dir is None:
            model = lgbm.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=es_rounds,
                eval_metric='logloss',
    #             verbose=-1)
                verbose=50)
        else:
            with open(f'{model_dir}/lgbm_fold{i}.pkl', 'rb') as f:
                model = pickle.load(f)

        pred = model.predict_proba(X_valid)
        oof[val_idx] = pred
        models.append(model)

        file = OUTPUT_DIR+f'lgbm_fold{i}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print()

    #cv = (oof.argmax(axis=-1) == y).mean()
    #print(f"CV-accuracy: {cv}")

    cv = f1_score(y, oof > 0.5, average="micro")
    print(f"CV-micro f1: {cv}")
    return oof, models


params = {
    'objective': "logloss",
    'learning_rate': 0.2,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,

    'max_depth': 7,
    'num_leaves': 35,
    'n_estimators': 1000000,
    "colsample_bytree": 0.9,
}

oof, models = fit_lgbm(train[features], train[[f"target_{i}" for i in range(10)]],
                       params=params, n_class=int(train["num_target"].max() + 1),
                       N_SPLITS=CFG.n_splits, folds=train["fold"].values)
"""
#print(oof.shape)
#np.save(OUTPUT_DIR+'oof.npy', oof)

print('finished')
