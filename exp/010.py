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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle


# https://www.kaggle.com/code/columbia2131/foursquare-iou-metrics
def get_id2poi(input_df: pd.DataFrame) -> dict:
    return dict(zip(input_df['id'], input_df['point_of_interest']))

def get_poi2ids(input_df: pd.DataFrame) -> dict:
    return input_df.groupby('point_of_interest')['id'].apply(set).to_dict()

def get_score(input_df: pd.DataFrame):
    scores = []
    for id_str, matches in zip(input_df['id'].to_numpy(), input_df['matches'].to_numpy()):
        targets = poi2ids[id2poi[id_str]]
        preds = set(matches.split())
        score = len((targets & preds)) / len((targets | preds))
        scores.append(score)
    scores = np.array(scores)
    return scores.mean()


def add_sep_token(df):
    # Before concatenation, fill NAN with unknown
    df.fillna('unknown', inplace = True)
    df['text'] = \
      df['near_name_0'] + '[SEP]' + df['near_address_0'] + '[SEP]' + df['near_city_0'] + '[SEP]' \
    + df['near_state_0'] + '[SEP]' + df['near_country_0'] + '[SEP]' + df['near_url_0'] + '[SEP]' + df['near_categories_0'] + '[SEP]' \
    + df['near_name_1'] + '[SEP]' + df['near_address_1'] + '[SEP]' + df['near_city_1'] + '[SEP]' \
    + df['near_state_1'] + '[SEP]' + df['near_country_1'] + '[SEP]' + df['near_url_1'] + '[SEP]' + df['near_categories_1'] + '[SEP]' \
    + df['near_name_2'] + '[SEP]' + df['near_address_2'] + '[SEP]' + df['near_city_2'] + '[SEP]' \
    + df['near_state_2'] + '[SEP]' + df['near_country_2'] + '[SEP]' + df['near_url_2'] + '[SEP]' + df['near_categories_2'] \
    + df['near_name_3'] + '[SEP]' + df['near_address_3'] + '[SEP]' + df['near_city_3'] + '[SEP]' \
    + df['near_state_3'] + '[SEP]' + df['near_country_3'] + '[SEP]' + df['near_url_3'] + '[SEP]' + df['near_categories_3'] + '[SEP]' \
    + df['near_name_4'] + '[SEP]' + df['near_address_4'] + '[SEP]' + df['near_city_4'] + '[SEP]' \
    + df['near_state_4'] + '[SEP]' + df['near_country_4'] + '[SEP]' + df['near_url_4'] + '[SEP]' + df['near_categories_4'] \
    + df['near_name_5'] + '[SEP]' + df['near_address_5'] + '[SEP]' + df['near_city_5'] + '[SEP]' \
    + df['near_state_5'] + '[SEP]' + df['near_country_5'] + '[SEP]' + df['near_url_5'] + '[SEP]' + df['near_categories_5'] + '[SEP]' \
    + df['near_name_6'] + '[SEP]' + df['near_address_6'] + '[SEP]' + df['near_city_6'] + '[SEP]' \
    + df['near_state_6'] + '[SEP]' + df['near_country_6'] + '[SEP]' + df['near_url_6'] + '[SEP]' + df['near_categories_6'] \
    + df['near_name_7'] + '[SEP]' + df['near_address_7'] + '[SEP]' + df['near_city_7'] + '[SEP]' \
    + df['near_state_7'] + '[SEP]' + df['near_country_7'] + '[SEP]' + df['near_url_7'] + '[SEP]' + df['near_categories_7'] + '[SEP]' \
    + df['near_name_8'] + '[SEP]' + df['near_address_8'] + '[SEP]' + df['near_city_8'] + '[SEP]' \
    + df['near_state_8'] + '[SEP]' + df['near_country_8'] + '[SEP]' + df['near_url_8'] + '[SEP]' + df['near_categories_8'] \
    + df['near_name_9'] + '[SEP]' + df['near_address_9'] + '[SEP]' + df['near_city_9'] + '[SEP]' \
    + df['near_state_9'] + '[SEP]' + df['near_country_9'] + '[SEP]' + df['near_url_9'] + '[SEP]' + df['near_categories_9']
    
    for i in range(10):
        del df[f'near_name_{i}'], df[f'near_address_{i}'], df[f'near_city_{i}'], df[f'near_state_{i}'], df[f'near_country_{i}'], df[f'near_url_{i}'], df[f'near_categories_{i}']
        gc.collect()
    return df


class CFG:
    EXP_ID = '010'
    seed = 71
    epochs = 3
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
    folds = [0, 1, 2]
    apex = True
    #model_name = 'xlm-roberta-base'
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #max_len = 180


import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


print('load data')
train = pd.read_csv('input/train_with_near_candidate_target_v2.csv')

for i in range(CFG.n_neighbors):
    train.loc[train[CFG.target]==train[f"near_target_{i}"], "target"] = i

train["target"] = train["target"].fillna(0)
train["target"] = train["target"].astype(int)

print('load features')

train = add_sep_token(train)

distance_features = unpickle('features/distance_features_v2.pkl')
features = list(distance_features.columns)

lat_lon_distance_features = unpickle('features/lon_lat_distaice_features.pkl')
features += list(lat_lon_distance_features.columns)

train = train[['text'] + [CFG.target, "target", "id"] + [f"near_id_{i}" for i in range(CFG.n_neighbors)]]

train = pd.concat([train, distance_features, lat_lon_distance_features], 1)
train[features] = train[features].astype(np.float16)

train.reset_index(drop=True, inplace=True)

kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(train, train["target"], train["target"]))):
    train.loc[val_idx, "fold"] = i

print(train[features].shape)

print(train.shape)
print(train.head())


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

    cv = (oof.argmax(axis=-1) == y).mean()
    print(f"CV-accuracy: {cv}")
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

oof, models = fit_lgbm(train[features], train["target"].astype(int),
                       params=params, n_class=int(train["target"].max() + 1),
                       N_SPLITS=CFG.n_splits, folds=train["fold"].values)

print(oof.shape)
np.save(OUTPUT_DIR+'oof.npy', oof)


near_ids = train[[f"near_id_{i}" for i in range(CFG.n_neighbors)]].values

matches = []
for id, ps, ids in tqdm(zip(train["id"], oof, near_ids)):
    idx = np.argmax(ps)
    if idx > 0 and ids[idx]==ids[idx]:
        matches.append(id + " " + ids[idx])
    else:
        matches.append(id)
train["matches"] = matches
print(f"CV: {get_score(train):.6f}")

