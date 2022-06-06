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


import Levenshtein
import difflib
import multiprocessing
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class CFG:
    seed = 46
    target = "point_of_interest"
    n_neighbors = 10
    n_splits = 3


random.seed(CFG.seed)
os.environ["PYTHONHASHSEED"] = str(CFG.seed)
np.random.seed(CFG.seed)


train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
test[CFG.target] = "TEST"


kf = GroupKFold(n_splits=2)
for i, (trn_idx, val_idx) in enumerate(kf.split(train, train[CFG.target], train[CFG.target])):
    train.loc[val_idx, "set"] = i
print(train["set"].value_counts())


## Parameters
NUM_NEIGHBOR = 20
SEED = 2022
THRESHOLD = 0.5
NUM_SPLIT = 10
feat_columns = ['dist', 'name', 'address', 'city', 
            'state', 'zip', 'url', 
           'phone', 'categories', 'country']
vec_columns = ['name', 'categories', 'address', 
               'state', 'url', 'country']
rec_columns = ['name', 'address', 'categories', 'address', 'phone']

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
seed_everything(SEED)


def LCS(S, T):
    dp = [[0] * (len(T) + 1) for _ in range(len(S) + 1)]
    for i in range(len(S)):
        for j in range(len(T)):
            dp[i + 1][j + 1] = max(dp[i][j] + (S[i] == T[j]), dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
    return dp[len(S)][len(T)]


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

# get manhattan distance
def manhattan(lat1, long1, lat2, long2):
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)

# get haversine distance
def vectorized_haversine(lats1, lats2, longs1, longs2):
    radius = 6371
    dlat=np.radians(lats2 - lats1)
    dlon=np.radians(longs2 - longs1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats1)) \
        * np.cos(np.radians(lats2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d

def recall_simple(df):
    threshold = 2

    val2id_d = {}
    for col in rec_columns:
        temp_df = df[['id', col]]
        temp_df[col] = temp_df[col].str.lower()
        val2id = temp_df.groupby(col)['id'].apply(set).to_dict()
        val2id_d[col] = val2id
        del val2id

    cus_ids = []
    match_ids = []
    for vals in tqdm(df[rec_columns + ['id']].fillna('null').values):
        cus_id = vals[-1]
        match_id = []

        rec_match_count = []
        for i in range(len(rec_columns)):
            col = rec_columns[i]

            if vals[i] != 'null':
                rec_match_count += list(val2id_d[col][vals[i].lower()])
        rec_match_count = dict(Counter(rec_match_count))

        for k, v in rec_match_count.items():
            if v > threshold:
                match_id.append(k)

        cus_ids += [cus_id] * len(match_id)
        match_ids += match_id

    train_df = pd.DataFrame()
    train_df['id'] = cus_ids
    train_df['match_id'] = match_ids
    train_df = train_df.drop_duplicates()
    del cus_ids, match_ids

    num_data = len(train_df)
    num_data_per_id = num_data / train_df['id'].nunique()
    print('Num of data: %s' % num_data)
    print('Num of data per id: %s' % num_data_per_id)

    return train_df


def recall_knn(df, Neighbors = 10):
    print('Start knn grouped by country')
    train_df_country = []
    for country, country_df in tqdm(df.groupby('country')):
        country_df = country_df.reset_index(drop = True)

        neighbors = min(len(country_df), Neighbors)
        knn = KNeighborsRegressor(n_neighbors = neighbors,
                                    metric = 'haversine',
                                    n_jobs = -1)
        knn.fit(country_df[['latitude','longitude']], country_df.index)
        dists, nears = knn.kneighbors(country_df[['latitude', 'longitude']],
                                        return_distance = True)

        for k in range(neighbors):
            cur_df = country_df[['id']]
            cur_df['match_id'] = country_df['id'].values[nears[:, k]]
            cur_df['kdist_country'] = dists[:, k]
            cur_df['kneighbors_country'] = k

            train_df_country.append(cur_df)
    train_df_country = pd.concat(train_df_country)

    print('Start knn')
    train_df = []
    knn = NearestNeighbors(n_neighbors = Neighbors)
    knn.fit(df[['latitude','longitude']], df.index)
    dists, nears = knn.kneighbors(df[['latitude','longitude']])

    for k in range(Neighbors):
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, k]]
        cur_df['kdist'] = dists[:, k]
        cur_df['kneighbors'] = k
        train_df.append(cur_df)

    train_df = pd.concat(train_df)
    train_df = train_df.merge(train_df_country,
                                 on = ['id', 'match_id'],
                                 how = 'outer')
    del train_df_country

    return train_df


def add_features(df):
    for col in tqdm(feat_columns):
        if col == 'dist':
            lat = data.loc[df['id']]['latitude'].values
            match_lat = data.loc[df['match_id']]['latitude'].values
            lon = data.loc[df['id']]['longitude'].values
            match_lon = data.loc[df['match_id']]['longitude'].values
            df['latdiff'] = (lat - match_lat)
            df['londiff'] = (lon - match_lon)
            df['manhattan'] = manhattan(lat, lon, match_lat, match_lon)
            df['euclidean'] = (df['latdiff'] ** 2 + df['londiff'] ** 2) ** 0.5
            df['haversine'] = vectorized_haversine(lat, match_lat, lon, match_lon)
            continue

        col_values = data.loc[df['id']][col].values.astype(str)
        matcol_values = data.loc[df['match_id']][col].values.astype(str)

        if col in vec_columns:
            tv_fit = tfidf_d[col]
            indexs = [id2index_d[i] for i in df['id']]
            match_indexs = [id2index_d[i] for i in df['match_id']]
            df[f'{col}_sim'] = tv_fit[indexs].multiply(tv_fit[match_indexs]).\
                                            sum(axis = 1).A.ravel()

        geshs = []
        levens = []
        jaros = []
        lcss = []
        for s, match_s in zip(col_values, matcol_values):
            if s != 'nan' and match_s != 'nan':
                geshs.append(difflib.SequenceMatcher(None, s, match_s).ratio())
                levens.append(Levenshtein.distance(s, match_s))
                jaros.append(Levenshtein.jaro_winkler(s, match_s))
                lcss.append(LCS(str(s), str(match_s)))
            else:
                geshs.append(np.nan)
                levens.append(np.nan)
                jaros.append(np.nan)
                lcss.append(np.nan)

        df[f'{col}_gesh'] = geshs
        df[f'{col}_leven'] = levens
        df[f'{col}_jaro'] = jaros
        df[f'{col}_lcs'] = lcss

        if col not in ['phone', 'zip']:
            df[f'{col}_len'] = list(map(len, col_values))
            df[f'match_{col}_len'] = list(map(len, matcol_values))
            df[f'{col}_len_diff'] = np.abs(df[f'{col}_len'] - df[f'match_{col}_len']) /\
                                        df[f'{col}_len']
            df[f'{col}_nleven'] = df[f'{col}_leven'] / \
                                    df[[f'{col}_len', f'match_{col}_len']].max(axis = 1)

            df[f'{col}_nlcsk'] = df[f'{col}_lcs'] / df[f'match_{col}_len']
            df[f'{col}_nlcs'] = df[f'{col}_lcs'] / df[f'{col}_len']

            df = df.drop(f'{col}_len', axis = 1)
            df = df.drop(f'match_{col}_len', axis = 1)
            gc.collect()

    return df


# special process
def get_lower(x):
    try:
        return x.lower()
    except:
        return x


df_train = []
for data in [train[train["set"]==0], train[train["set"]==1]]:
    for col in data.columns:
        if data[col].dtype == object and col != 'id':
            data[col] = data[col].apply(get_lower)

    id2index_d = dict(zip(data['id'].values, data.index))

    tfidf_d = {}
    for col in vec_columns:
        tfidf = TfidfVectorizer()
        tv_fit = tfidf.fit_transform(data[col].fillna('nan'))
        tfidf_d[col] = tv_fit

    train_data_simple = recall_simple(data)
    train_data = recall_knn(data, NUM_NEIGHBOR)

    print('train data by knn: %s' % len(train_data))
    train_data = train_data.merge(train_data_simple,
                                 on = ['id', 'match_id'],
                                 how = 'outer')
    del train_data_simple; gc.collect()
    df_train.append(train_data)
    del train_data; gc.collect()

df_train = pd.concat(df_train, 0).reset_index(drop=True) 

id2poi = dict(zip(train['id'].values, train['point_of_interest'].values))

df_train['target'] = (df_train['id'].map(id2poi) == df_train['match_id'].map(id2poi)).astype(int)


print(df_train.shape)
print(df_train['target'].value_counts())

#df_train = df_train[df_train['id']!=df_train['match_id']].reset_index(drop=True)
#print(df_train.shape)
#print(df_train['target'].value_counts())

df_train.to_csv('input/train_pairs.csv', index=False)


