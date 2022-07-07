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


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

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


longest_substring_columns = [
    'name',
    'address',
    'categories',
]

# source: https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
@numba.jit(nopython=True, nogil=True, cache=True)
def longestSubstringFinder(string1: str, string2: str):
    answer = 0
    len1, len2 = len(string1), len(string2)

    for i in range(len1):
        for j in range(len2):
            lcs_temp = 0
            match = 0
            while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and string1[i+lcs_temp] == string2[j+lcs_temp]):
                match += 1
                lcs_temp += 1
            if match > answer:
                answer = match
    return np.uint8(answer)


# Longest substring feature
def add_longest_substr(df):
    for col in longest_substring_columns:
        df[f'{col}_longest_substr'] = df[[f'{col}_1', f'{col}_2']].apply(lambda args: longestSubstringFinder(*args), axis=1, raw=True).astype(np.uint8)
        df[f'{col}_longest_substr_ratio'] = (
                (df[f'{col}_longest_substr'] * 2) / (df[f'{col}_1'].apply(len) + df[f'{col}_2'].apply(len))
            ).astype(np.float32)
    return df


class CFG:
    EXP_ID = '069'
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
TRAIN_FEATURES = [
                'kdist',
                'kneighbors',
                #'kdist_country',
                #'kneighbors_country',
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

                #'name_sim_use',
                #'categories_sim_use',
                'text_sim_w2v',

                'name_longest_substr_ratio',
                'address_longest_substr_ratio',
                'categories_longest_substr_ratio',

                'text_sim_bm25_svd',
                'text_sim_mpnet',

                'latitude_round_1',
                'longitude_round_1',
                'latitude_round_2',
                'longitude_round_2',
                'same_number',  
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
id_2_address = {k:v for k, v in zip(data['id'].values, data['address'].fillna('noaddress').values)}

del data;gc.collect()

#categories_2_categories_use_vector = unpickle('features/categories_2_categories_use_vector.pkl')
#name_2_name_use_vector = unpickle('features/name_2_name_use_vector.pkl')

print('load data')
train = pd.read_csv('input/downsampled_with_oof_059_train_data.csv')
train1= pd.read_csv('input/downsampled_with_oof_062_valid_data.csv')

train = pd.concat([train, train1], 0).reset_index(drop=True)
del train1; gc.collect()
print(train['label'].value_counts())

train['latitude_1'] = train['id'].map(id_2_lat)
train['latitude_2'] = train['match_id'].map(id_2_lat)
train['longitude_1'] = train['id'].map(id_2_lon)
train['longitude_2'] = train['match_id'].map(id_2_lon)
train['latitude_round_1'] = train['latitude_1'].round(1)
train['longitude_round_1'] = train['longitude_1'].round(1)
train['latitude_round_2'] = train['latitude_2'].round(1)
train['longitude_round_2'] = train['longitude_2'].round(1)
train = add_haversine_distance(train)
del train['latitude_1'], train['latitude_2'], train['longitude_1'], train['longitude_2']; gc.collect()


id_2_w2v_vec_train = unpickle(f'features/id_2_text_w2v_vector_50d_train_ids.pkl')
sim = []
for nv1, nv2 in tqdm(zip(train['id'].map(id_2_w2v_vec_train), train['match_id'].map(id_2_w2v_vec_train))):
    sim.append(cosine_similarity(nv1, nv2))
train['text_sim_w2v'] = sim
del id_2_w2v_vec_train, sim; gc.collect()

id_2_text_tfidf_bm25_svd_vector = unpickle(f'features/id_2_text_tfidf_bm25_svd_vector_64d_train_ids.pkl')
sim = []
for nv1, nv2 in tqdm(zip(train['id'].map(id_2_text_tfidf_bm25_svd_vector), train['match_id'].map(id_2_text_tfidf_bm25_svd_vector))):
    sim.append(cosine_similarity(nv1, nv2))
train['text_sim_bm25_svd'] = sim
del id_2_text_tfidf_bm25_svd_vector, sim; gc.collect()

id_2_text_mpnet_vector = unpickle(f'features/id_2_text_mpnet_vector_train_ids.pkl')
sim = []
for nv1, nv2 in tqdm(zip(train['id'].map(id_2_text_mpnet_vector), train['match_id'].map(id_2_text_mpnet_vector))):
    sim.append(cosine_similarity(nv1, nv2))
train['text_sim_mpnet'] = sim
del id_2_text_mpnet_vector, sim; gc.collect()

id_2_text = unpickle('features/id_2_text.pkl')

train['text_1'] = train['id'].map(id_2_text)
train['text_2'] = train['match_id'].map(id_2_text)

train['name_1'] = train['id'].map(id_2_name)
train['name_2'] = train['match_id'].map(id_2_name)

train['categories_1'] = train['id'].map(id_2_cat)
train['categories_2'] = train['match_id'].map(id_2_cat)
train["category_venn"] = train[["categories_1", "categories_2"]] \
        .progress_apply(lambda row: categorical_similarity(row.categories_1, row.categories_2),
                        axis=1)

train['address_1'] = train['id'].map(id_2_address)
train['address_2'] = train['match_id'].map(id_2_address)
train['number_1'] = train['address_1'].str.extract('(\d+)')
train['number_2'] = train['address_2'].str.extract('(\d+)')
train['same_number'] = (train['number_1'] == train['number_2']).astype(int)
del train['number_1'], train['number_2']; gc.collect()

train = add_longest_substr(train)
del train['address_1'], train['address_2']; gc.collect()
"""
use_sim = []
for nv1, nv2 in tqdm(zip(train['name_1'].map(name_2_name_use_vector), train['name_2'].map(name_2_name_use_vector))):
    use_sim.append(cosine_similarity(nv1, nv2))
train['name_sim_use'] = use_sim
del train['name_1'], train['name_2'], use_sim; gc.collect()

categories_sim = []
for nv1, nv2 in tqdm(zip(train['categories_1'].map(categories_2_categories_use_vector), train['categories_2'].map(categories_2_categories_use_vector))):
    categories_sim.append(cosine_similarity(nv1, nv2))
train['categories_sim_use'] = categories_sim
del train['categories_1'], train['categories_2'], categories_sim; gc.collect()
"""
print(train.shape)
print(train['label'].value_counts())
print(train[TRAIN_FEATURES].shape)


from catboost import CatBoostRegressor, CatBoostClassifier, Pool

text_cols = ['text_1', 'text_2']
#categorical_cols = [f'{v}_1' for v in cat_columns] + [f'{v}_2' for v in cat_columns]

def fit_cat(X, y, params=None,
        es_rounds=20,
        seed=42, N_SPLITS=5,
        n_class=None, model_dir=None, folds=None):

    train_pool = Pool(
            X,
            y,
            #cat_features=categorical_cols,
            text_features=text_cols,
            #feature_names=list(X_tr)
    )

    if model_dir is None:
        model = CatBoostClassifier(**params)
        model.fit(
                train_pool,
                verbose=100
        )
    else:
        with open(f'{model_dir}/cat_fold{i}.pkl', 'rb') as f:
            model = pickle.load(f)

    file = OUTPUT_DIR+f'cat_all_train.pkl'
    pickle.dump(model, open(file, 'wb'))
    print()


params = {
    'objective': "Logloss", # "RMSE", # "MultiClass",
    'learning_rate': 0.2,
    #'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,

    #'max_depth': 7,
    #'num_leaves': 35,
    'n_estimators': 10000,
    #"colsample_bytree": 0.9,
    #'use_best_model': True,
    #'cat_features': ['country'],
    #'text_features': ['categories'],
    'task_type': "GPU",
}


fit_cat(train[TRAIN_FEATURES], train["label"].astype(int),
                       params=params, n_class=int(train["label"].max() + 1),
                       N_SPLITS=CFG.n_splits)


