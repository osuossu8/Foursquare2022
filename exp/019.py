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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold
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


def post_process(df):
    id2match = dict(zip(df['id'].values, df['matches'].str.split()))

    for base, match in df[['id', 'matches']].values:
        match = match.split()
        if len(match) == 1:
            continue

        for m in match:
            if base not in id2match[m]:
                id2match[m].append(base)
    df['matches'] = df['id'].map(id2match).map(' '.join)
    return df


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
    EXP_ID = '019'
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


NUM_NEIGHBOR = 20
SEED = 2022
THRESHOLD = 0.5
NUM_SPLIT = 5
TRAIN_FEATURES = ['kdist',
                'kneighbors',
                'kdist_country',
                'kneighbors_country',
                'name_sim',
                'name_gesh',
                'name_leven',
                'name_jaro',
                'name_lcs',
                'name_len_diff',
                'name_nleven',
                'name_nlcsk',
                'name_nlcs',
                'address_sim',
                'address_gesh',
                'address_leven',
                'address_jaro',
                'address_lcs',
                'address_len_diff',
                'address_nleven',
                'address_nlcsk',
                'address_nlcs',
                'city_gesh',
                'city_leven',
                'city_jaro',
                'city_lcs',
                'city_len_diff',
                'city_nleven',
                'city_nlcsk',
                'city_nlcs',
                'state_sim',
                'state_gesh',
                'state_leven',
                'state_jaro',
                'state_lcs',
                'state_len_diff',
                'state_nleven',
                'state_nlcsk',
                'state_nlcs',
                'zip_gesh',
                'zip_leven',
                'zip_jaro',
                'zip_lcs',
                'url_sim',
                'url_gesh',
                'url_leven',
                'url_jaro',
                'url_lcs',
                'url_len_diff',
                'url_nleven',
                'url_nlcsk',
                'url_nlcs',
                'phone_gesh',
                'phone_leven',
                'phone_jaro',
                'phone_lcs',
                'categories_sim',
                'categories_gesh',
                'categories_leven',
                'categories_jaro',
                'categories_lcs',
                'categories_len_diff',
                'categories_nleven',
                'categories_nlcsk',
                'categories_nlcs',
                'country_sim',
                'country_gesh',
                'country_leven',
                'country_nleven',]


import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


print('load data')
train1 = pd.read_csv('input/train_data1.csv')
print(train1['label'].value_counts())

train2 = pd.read_csv('input/train_data2.csv')
print(train2['label'].value_counts())

train3 = pd.read_csv('input/train_data3.csv')
print(train3['label'].value_counts())

train4 = pd.read_csv('input/train_data4.csv')
print(train4['label'].value_counts())

train5 = pd.read_csv('input/train_data5.csv')
print(train5['label'].value_counts())

train = pd.concat([
    train1, train2, train3, train4, train5
], 0).reset_index(drop=True)

del train1, train2, train3, train4, train5; gc.collect()

print(train.shape)
print(train['label'].value_counts())
print(train[TRAIN_FEATURES].shape)
print(train[TRAIN_FEATURES].head())


#sys.exit()


#kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
#kf = StratifiedGroupKFold(n_splits=CFG.n_splits)
kf = GroupKFold(n_splits=CFG.n_splits)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(X=train, y=train["label"], groups=train["id"]))):
    train.loc[val_idx, "fold"] = i

for i in range(CFG.n_splits):
    print()
    print(train.query(f'fold == {i}')['label'].value_counts())
    print(train.query(f'fold == {i}')['id'].nunique())
    print()


import catboost
# from catboost import CatBoostClassifier, Pool

def fit_cat(X, y, params=None, es_rounds=20, seed=42, N_SPLITS=5,
             n_class=None, model_dir=None, folds=None):
    models = []
    #oof = np.zeros((len(y), n_class), dtype=np.float64)
    oof = np.zeros((len(y)), dtype=np.float64)

    for i in tqdm(range(CFG.n_splits)):
        logger.info(f"== fold {i} ==")
        trn_idx = folds!=i
        val_idx = folds==i
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        if model_dir is None:
            #model = catboost.CatBoostClassifier(**params)
            model = catboost.CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                #cat_features=categorical_features,
                #text_features=['text'],
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=es_rounds,
                # eval_metric='logloss',
    #             verbose=-1)
                verbose=100
            )
        else:
            with open(f'{model_dir}/cat_fold{i}.pkl', 'rb') as f:
                model = pickle.load(f)

        #pred = model.predict_proba(X_valid)[:, 1]
        pred = model.predict(X_valid)
        oof[val_idx] = pred
        models.append(model)

        file = OUTPUT_DIR+f'cat_fold{i}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print()

    cv = (oof.argmax(axis=-1) == y).mean()
    #cv = ((oof > 0.5) == y).mean()
    logger.info(f"CV-accuracy: {cv}")
    return oof, models


params = {
    'objective': "RMSE", # "MultiClass", # "Logloss",
    'learning_rate': 0.2,
    #'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,

    #'max_depth': 7,
    #'num_leaves': 35,
    'n_estimators': 1000000,
    #"colsample_bytree": 0.9,
    'use_best_model': True,
    #'cat_features': ['country'],
    #'text_features': ['categories'],
    'task_type': "GPU",

    #'class_weights': {0:1, 1:11}
}


oof, models = fit_cat(train[TRAIN_FEATURES], train["label"].astype(int),
                       params=params, n_class=int(train["label"].max() + 1),
                       N_SPLITS=CFG.n_splits, folds=train["fold"].values)


print(oof.shape)
# np.save(OUTPUT_DIR+'oof.npy', oof)
"""
models = [unpickle(OUTPUT_DIR+f'cat_fold{i}.pkl') for i in range(3)]

test_pos = pd.read_csv('input/train_pairs_set_0_target_1.csv')
test_neg = pd.read_csv('input/train_pairs_set_0_target_0.csv')

test = pd.concat([test_pos, test_neg], 0).reset_index(drop=True)

del test_pos, test_neg; gc.collect()

#test['pred'] = np.mean([cat_model.predict(test[TRAIN_FEATURES]) for cat_model in models], 0)
test['pred'] = np.mean([cat_model.predict_proba(test[TRAIN_FEATURES])[:, 1] for cat_model in models], 0)
test = test[test['pred'] > 0.5][['id', 'match_id']]
# test = test[test['pred'] > 0.5][['id', 'match_id']]
print(test['id'].nunique())

test = test.groupby('id')['match_id'].apply(list).reset_index()
test['matches'] = test['match_id'].apply(lambda x: ' '.join(set(x)))
test['matches'] = test['matches']+' '+test['id']
test['matches'] = test['matches'].map(lambda x: ' '.join(set(x.split())))

train = pd.read_csv('input/train.csv', usecols=['id', 'point_of_interest'])
test = pd.merge(test, train, on='id', how='inner')

del train; gc.collect()

print(test[['id', 'matches']].head(10))

# test = post_process(test)

#train = pd.read_csv('input/train.csv')

id2poi = get_id2poi(test)
poi2ids = get_poi2ids(test)

#del train; gc.collect()

logger.info(f"IoU: {get_score(test):.6f}")


out_df = pd.DataFrame()
out_df['id'] = test_data['id'].unique().tolist()
out_df['match_id'] = out_df['id']


## Prediction
count = 0
start_row = 0
pred_df = pd.DataFrame()
unique_id = test_data['id'].unique().tolist()
num_split_id = len(unique_id) // NUM_SPLIT
for k in range(1, NUM_SPLIT + 1):
    print('Current split: %s' % k)
    end_row = start_row + num_split_id
    if k < NUM_SPLIT:
        cur_id = unique_id[start_row : end_row]
        cur_data = test_data[test_data['id'].isin(cur_id)]
    else:
        cur_id = unique_id[start_row: ]
        cur_data = test_data[test_data['id'].isin(cur_id)]
    
    # add features & model prediction
    #cur_data = add_features(cur_data)

    # cur_data_cat = Pool(cur_data[TRAIN_FEATURES])
    cur_data_cat = cur_data[TRAIN_FEATURES]
    cur_data['pred'] = 0

    for fold in range(CFG.n_splits):
        cat_model = unpickle(OUTPUT_DIR+f'cat_fold{fold}.pkl')
        #cur_data['pred'] += cat_model.predict(cur_data_cat)/CFG.n_splits
        cur_data['pred'] += cat_model.predict(cur_data_cat)/CFG.n_splits

    cur_pred_df = cur_data[cur_data['pred'] > 0][['id', 'match_id']]
    pred_df = pd.concat([pred_df, cur_pred_df])
    
    start_row = end_row
    count += len(cur_data)

    del cur_data, cur_pred_df
    gc.collect()
print(count)

out_df = pd.concat([out_df, pred_df])
out_df = out_df.groupby('id')['match_id'].\
                        apply(list).reset_index()
out_df['matches'] = out_df['match_id'].apply(lambda x: ' '.join(set(x)))

print(out_df['id'].nunique())

#out_df = post_process(out_df)
#print('Unique id: %s' % len(out_df))
print(out_df.head(10))

#out_df['matches'] = out_df['matches']+' '+out_df['id']
#out_df['matches'] = out_df['matches'].map(lambda x: ' '.join(set(x.split())))

train = pd.read_csv('input/train.csv')

id2poi = get_id2poi(train)
poi2ids = get_poi2ids(train)

del train; gc.collect()

logger.info(f"IoU: {get_score(out_df):.6f}")
"""
