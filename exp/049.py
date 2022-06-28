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
tqdm.pandas()

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


def categorical_similarity(A, B):
    if not A or not B:
        return -1

    A = set(str(A).split(", "))
    B = set(str(B).split(", "))

    # Find intersection of two sets
    nominator = A.intersection(B)

    similarity_1 = len(nominator) / len(A)
    similarity_2 = len(nominator) / len(B)

    return max(similarity_1, similarity_2)


def name_similarity(A, B):
    if not A or not B:
        return -1

    A = set(str(A).split(" "))
    B = set(str(B).split(" "))

    # Find intersection of two sets
    nominator = A.intersection(B)

    similarity_1 = len(nominator) / len(A)
    similarity_2 = len(nominator) / len(B)

    return max(similarity_1, similarity_2)


EARTH_RADIUS = 6371

import numba
# Numba optimized haversine distance
@numba.jit(nopython=True)
def haversine_np(args):
    lon1, lat1, lon2, lat2 = args
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = EARTH_RADIUS * c
    return km


# Adds haversine distance between two points
def add_haversine_distance(df):
    df['haversine_distance'] = np.apply_along_axis(
            haversine_np, 1,
            df[['longitude_1', 'latitude_1', 'longitude_2', 'latitude_2']].values.astype(np.float32)
        ).astype(np.float32)
    return df


def manhattan(lat1, long1, lat2, long2):
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)


class CFG:
    EXP_ID = '049'
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
    apex = True
    #model_name = 'xlm-roberta-base'
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #max_len = 180

               

NUM_NEIGHBOR = 25
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
                'country_nleven',
                
                'text_1',
                'text_2',

                'category_venn',
                'haversine_distance',

                'name_sim_use',
                'categories_sim_use',

                #'latdiff',
                #'londiff',
                #'manhattan',
                #'euclidean',

                'text_sim_w2v',
                'name_venn',
]

# Optimized cosine similarity function
#@numba.jit(nopython=True)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


data = pd.read_csv("input/train.csv")

id_2_cat = {k:v for k, v in zip(data['id'].values, data['categories'].fillna('nocategories').values)}
id_2_name = {k:v for k, v in zip(data['id'].values, data['name'].fillna('noname').values)}
id_2_lat = {k:v for k, v in zip(data['id'].values, data['latitude'].values)}
id_2_lon = {k:v for k, v in zip(data['id'].values, data['longitude'].values)}

del data;gc.collect()

categories_2_categories_use_vector = unpickle('features/categories_2_categories_use_vector.pkl')
name_2_name_use_vector = unpickle('features/name_2_name_use_vector.pkl')

print('load data')
train = pd.read_csv('input/downsampled_with_oof_037_train_data.csv')
#train = train[train['oof']>0.01].reset_index(drop=True)
print(train['label'].value_counts())

train['latitude_1'] = train['id'].map(id_2_lat)
train['latitude_2'] = train['match_id'].map(id_2_lat)

train['longitude_1'] = train['id'].map(id_2_lon)
train['longitude_2'] = train['match_id'].map(id_2_lon)

#train['latdiff'] = train['latitude_1'] - train['latitude_2']
#train['londiff'] = train['longitude_1'] - train['longitude_2']

#train['euclidean'] = (train['latdiff'] ** 2 + train['londiff'] ** 2) ** 0.5
#train['manhattan'] = manhattan(train['latitude_1'], train['longitude_1'], train['latitude_2'], train['longitude_2'])

train = add_haversine_distance(train)

id_2_w2v_vec_train = unpickle(f'features/id_2_text_w2v_vector_50d_train_ids.pkl')
w2v_sim = []
for nv1, nv2 in tqdm(zip(train['id'].map(id_2_w2v_vec_train), train['match_id'].map(id_2_w2v_vec_train))):
    w2v_sim.append(cosine_similarity(nv1, nv2))
train['text_sim_w2v'] = w2v_sim
del id_2_w2v_vec_train; gc.collect()

id_2_text = unpickle('features/id_2_text.pkl')

train['text_1'] = train['id'].map(id_2_text)
train['text_2'] = train['match_id'].map(id_2_text)

train['name_1'] = train['id'].map(id_2_name)
train['name_2'] = train['match_id'].map(id_2_name)
train["name_venn"] = train[["name_1", "name_2"]] \
        .progress_apply(lambda row: name_similarity(row.name_1, row.name_2),
                        axis=1)

train['categories_1'] = train['id'].map(id_2_cat)
train['categories_2'] = train['match_id'].map(id_2_cat)
train["category_venn"] = train[["categories_1", "categories_2"]] \
        .progress_apply(lambda row: categorical_similarity(row.categories_1, row.categories_2),
                        axis=1)

use_sim = []
for nv1, nv2 in tqdm(zip(train['name_1'].map(name_2_name_use_vector), train['name_2'].map(name_2_name_use_vector))):
    use_sim.append(cosine_similarity(nv1, nv2))
train['name_sim_use'] = use_sim

categories_sim = []
for nv1, nv2 in tqdm(zip(train['categories_1'].map(categories_2_categories_use_vector), train['categories_2'].map(categories_2_categories_use_vector))):
    categories_sim.append(cosine_similarity(nv1, nv2))
train['categories_sim_use'] = categories_sim

print(train.shape)
print(train['label'].value_counts())
print(train[TRAIN_FEATURES].shape)
# print(train[TRAIN_FEATURES].head())


print(train[['name_venn', 'text_sim_w2v', 'name_sim_use', 'categories_sim_use', 'category_venn']].head())


#kf = StratifiedGroupKFold(n_splits=CFG.n_splits)
kf = GroupKFold(n_splits=CFG.n_splits)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(X=train, y=train["label"], groups=train["id"]))):
    train.loc[val_idx, "fold"] = i

for i in range(CFG.n_splits):
    print()
    print(train.query(f'fold == {i}')['label'].value_counts())
    print(train.query(f'fold == {i}')['id'].nunique())
    print()


from catboost import CatBoostRegressor, CatBoostClassifier, Pool

text_cols = ['text_1', 'text_2']
#categorical_cols = [f'{v}_1' for v in cat_columns] + [f'{v}_2' for v in cat_columns]

def fit_cat(X, y, params=None, 
        es_rounds=20, 
        seed=42, N_SPLITS=5,
        n_class=None, model_dir=None, folds=None):
    models = []
    oof = np.zeros((len(y)), dtype=np.float16)

    for i in tqdm(range(CFG.n_splits)):
        logger.info(f"== fold {i} ==")
        trn_idx = folds!=i
        val_idx = folds==i
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        train_pool = Pool(
            X_train, 
            y_train, 
            #cat_features=categorical_cols,
            text_features=text_cols,
            #feature_names=list(X_tr)
        )
        valid_pool = Pool(
            X_valid, 
            y_valid, 
            #cat_features=categorical_cols,
            text_features=text_cols,
            #feature_names=list(X_tr)
        )

        if model_dir is None:
            model = CatBoostClassifier(**params)
            #model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=es_rounds,
                # eval_metric='logloss',
                verbose=100
            )
        else:
            with open(f'{model_dir}/cat_fold{i}.pkl', 'rb') as f:
                model = pickle.load(f)

        pred = model.predict_proba(X_valid)[:, 1]
        oof[val_idx] = pred
        models.append(model)

        file = OUTPUT_DIR+f'cat_fold{i}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print()

    cv = ((oof > 0.5) == y).mean()
    logger.info(f"CV-accuracy: {cv}")
    return oof, models


params = {
    'objective': "Logloss", # "RMSE", # "MultiClass",
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
}


oof, models = fit_cat(train[TRAIN_FEATURES], train["label"].astype(int),
                       params=params, n_class=int(train["label"].max() + 1),
                       N_SPLITS=CFG.n_splits, folds=train["fold"].values)

print(oof.shape)
np.save(OUTPUT_DIR+'oof.npy', oof)

id_2_text = unpickle('features/id_2_text.pkl')
id_2_w2v_vec_valid = unpickle(f'features/id_2_text_w2v_vector_50d_valid_ids.pkl')

models = [unpickle(OUTPUT_DIR+f'cat_fold{i}.pkl') for i in range(CFG.n_splits)]

res_df = []
for test_path in tqdm([
    'input/valid_data_candidate_25_1.csv',
    'input/valid_data_candidate_25_2.csv',
    'input/valid_data_candidate_25_3.csv',
    'input/valid_data_candidate_25_4.csv',
    'input/valid_data_candidate_25_5.csv'
    ]):

    test = pd.read_csv(test_path)

    test['latitude_1'] = test['id'].map(id_2_lat)
    test['latitude_2'] = test['match_id'].map(id_2_lat)

    test['longitude_1'] = test['id'].map(id_2_lon)
    test['longitude_2'] = test['match_id'].map(id_2_lon)

    #test['latdiff'] = test['latitude_1'] - test['latitude_2']
    #test['londiff'] = test['longitude_1'] - test['longitude_2']

    #test['euclidean'] = (test['latdiff'] ** 2 + test['londiff'] ** 2) ** 0.5
    #test['manhattan'] = manhattan(test['latitude_1'], test['longitude_1'], test['latitude_2'], test['longitude_2'])

    test = add_haversine_distance(test)

    w2v_sim = []
    for nv1, nv2 in tqdm(zip(test['id'].map(id_2_w2v_vec_valid), test['match_id'].map(id_2_w2v_vec_valid))):
        w2v_sim.append(cosine_similarity(nv1, nv2))
    test['text_sim_w2v'] = w2v_sim

    test['text_1'] = test['id'].map(id_2_text)
    test['text_2'] = test['match_id'].map(id_2_text)

    test['name_1'] = test['id'].map(id_2_name)
    test['name_2'] = test['match_id'].map(id_2_name)
    test["name_venn"] = test[["name_1", "name_2"]] \
        .progress_apply(lambda row: name_similarity(row.name_1, row.name_2),
                        axis=1)

    test['categories_1'] = test['id'].map(id_2_cat)
    test['categories_2'] = test['match_id'].map(id_2_cat)
    test["category_venn"] = test[["categories_1", "categories_2"]] \
        .progress_apply(lambda row: categorical_similarity(row.categories_1, row.categories_2),
                        axis=1)

    use_sim = []
    for nv1, nv2 in tqdm(zip(test['name_1'].map(name_2_name_use_vector), test['name_2'].map(name_2_name_use_vector))):
        use_sim.append(cosine_similarity(nv1, nv2))
    test['name_sim_use'] = use_sim

    categories_sim = []
    for nv1, nv2 in tqdm(zip(test['categories_1'].map(categories_2_categories_use_vector), test['categories_2'].map(categories_2_categories_use_vector))):
        categories_sim.append(cosine_similarity(nv1, nv2))
    test['categories_sim_use'] = categories_sim

    test['pred'] = 0
    for cat_model in tqdm(models):

        test['pred'] += cat_model.predict_proba(test[TRAIN_FEATURES])[:, 1]/len(models)

    print(test[['id', 'match_id', 'pred']])
    res_df.append(test[test['pred'] > 0.5][['id', 'match_id']])

    del test; gc.collect()

test = pd.concat(res_df, 0).reset_index(drop=True)

del res_df; gc.collect()
del id_2_text; gc.collect()
del id_2_w2v_vec_valid; gc.collect()

print(test['id'].nunique())

test = test.groupby('id')['match_id'].apply(list).reset_index()
print(test.head())

test['matches'] = test['match_id'].apply(lambda x: ' '.join(set(x)))
test['matches'] = test['matches']+' '+test['id']
test['matches'] = test['matches'].map(lambda x: ' '.join(set(x.split())))

train = pd.read_csv('input/train.csv', usecols=['id', 'point_of_interest'])
test = pd.merge(test, train, on='id', how='inner')

del train; gc.collect()

#test = post_process(test)

print(test[['id', 'matches']].head(10))

id2poi = get_id2poi(test)
poi2ids = get_poi2ids(test)

logger.info(f"IoU: {get_score(test):.6f}")

