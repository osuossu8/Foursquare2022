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

from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle
logger = init_logger(log_file='log/' + f"create_candidate_25_faiss_v3_valid.log")

import faiss
from sentence_transformers import SentenceTransformer
from cuml.neighbors import NearestNeighbors as NearestNeighborsGPU

tqdm.pandas()


def get_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# https://www.kaggle.com/code/guoyonfan/training-data-for-binary-lgb-baseline-0-834/notebook
from gensim.models import word2vec
from gensim.models import KeyedVectors


class W2VVectorizer:
    def __init__(self, sentences, vec_size):
        self.sentences = sentences.apply(lambda x: x.split())
        self.vec_size = vec_size

        print('fit models ...')
        self.model = word2vec.Word2Vec(self.sentences, 
                                       vector_size=self.vec_size,
                                       min_count=1,
                                       window=1,
                                       epochs=100)
        
    def vectorize(self, word_list : list) -> np.array:
        V = []
        for word in word_list:
            try:
                vector = self.model.wv[word]
            except:
                vector = [j for j in range(self.vec_size)]
            V.append(vector)
        return np.mean(V, 0)        

    def get_vectors(self) -> pd.DataFrame:

        print('get vectors ...')
        vectors = self.sentences.progress_apply(lambda x: self.vectorize(x))
        vectors = np.stack(vectors, 0)
        return vectors

## Parameters
is_debug = False
SEED = 2022
num_neighbors = 25 # 20
num_split = 5
feat_columns = ['name', 'address', 'city', 
            'state', 'zip', 'url', 
           'phone', 'categories', 'country']
vec_columns = ['name', 'categories', 'address', 
               'state', 'url', 'country']

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


def recall_knn(df, Neighbors = 10):
    logger.info('Start knn')
    train_df = []
    knn = NearestNeighbors(n_neighbors = Neighbors)
    knn.fit(df[['latitude','longitude']], df.index)
    dists, nears = knn.kneighbors(df[['latitude','longitude']])

    logger.info('Start faiss')
    embedder_mpnet = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    batches = list(get_batches(df['name'].fillna('noname').values, 1000))
    EMBEDDINGS_MPNET = np.zeros(shape=[len(df['name']), 768], dtype=np.float32)
    OFFSET = 0
    for batch in tqdm(batches):
        n = len(batch)
        EMBEDDINGS_MPNET[OFFSET:OFFSET + n] = embedder_mpnet.encode(batch, show_progress_bar=False).astype(np.float32)
        OFFSET += n

    del embedder_mpnet, batches; gc.collect()
    print(EMBEDDINGS_MPNET.shape)

    res = faiss.StandardGpuResources()
    dim = 768
    nlist = 1024
    M = 32
    nbits = 8
    metric = faiss.METRIC_L2
    ivfpq_config = faiss.GpuIndexIVFPQConfig()
    ivfpq_config.usePrecomputedTables = True

    faiss_index = faiss.GpuIndexIVFPQ(res, dim, nlist, M, nbits, metric, ivfpq_config)
    faiss_index.train(EMBEDDINGS_MPNET)
    faiss_index.add(EMBEDDINGS_MPNET)

    faiss_index.nprobe = 5
    dists2, nears2 = faiss_index.search(EMBEDDINGS_MPNET, 25)
    del faiss_index, EMBEDDINGS_MPNET; gc.collect()

    df['text'] = ''
    for v in vec_columns:
        df['text'] += df[v].fillna('nan') + ' '

    wv2_vec = W2VVectorizer(df['text'], vec_size=50).get_vectors()

    print('Start knn with w2v text emb')
    knn = NearestNeighbors(n_neighbors = Neighbors)
    knn.fit(wv2_vec, df.index)
    dists3, nears3 = knn.kneighbors(wv2_vec)

    for k in range(Neighbors):            
        cur_df = df[['id']]
        cur_df['match_id'] = df['id'].values[nears[:, k]]
        cur_df['kdist'] = dists[:, k]
        cur_df['kneighbors'] = k
        cur_df = cur_df[(cur_df['id']!=cur_df['match_id'])&(cur_df['kdist']<5)]
        train_df.append(cur_df)
    
        cur_df2 = df[['id']]
        cur_df2['match_id'] = df['id'].values[nears2[:, k]]
        cur_df2['kdist'] = dists2[:, k]
        cur_df2['kneighbors'] = k
        cur_df2 = cur_df2[(cur_df2['id']!=cur_df2['match_id'])&(cur_df2['kdist']<5)]
        train_df.append(cur_df2)

        cur_df3 = df[['id']]
        cur_df3['match_id'] = df['id'].values[nears3[:, k]]
        cur_df3['kdist'] = dists3[:, k]
        cur_df3['kneighbors'] = k
        cur_df3 = cur_df3[(cur_df3['id']!=cur_df3['match_id'])&(cur_df3['kdist']<5)]
        train_df.append(cur_df3)

    train_df = pd.concat(train_df)
    train_df = reduce_mem_usage(train_df)
    train_df = train_df.drop_duplicates(subset=['id', 'match_id'], keep='first').reset_index(drop=True)
    train_df['id'] = train_df['id'].astype('object')
    train_df['match_id'] = train_df['match_id'].astype('object')
    return train_df


def add_features(df):
    for col in tqdm(feat_columns):
        if col in vec_columns:
            tv_fit = tfidf_d[col]
            indexs = [id2index_d[i] for i in df['id']]
            match_indexs = [id2index_d[i] for i in df['match_id']]
            df[f'{col}_sim'] = tv_fit[indexs].multiply(tv_fit[match_indexs]).sum(axis = 1).A.ravel()

        col_values = data.loc[df['id']][col].values.astype(str)
        matcol_values = data.loc[df['match_id']][col].values.astype(str)

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
            df[f'{col}_len_diff'] = np.abs(df[f'{col}_len'] - df[f'match_{col}_len'])
            df[f'{col}_nleven'] = df[f'{col}_leven'] / \
                                    df[[f'{col}_len', f'match_{col}_len']].max(axis = 1)

            df[f'{col}_nlcsk'] = df[f'{col}_lcs'] / df[f'match_{col}_len']
            df[f'{col}_nlcs'] = df[f'{col}_lcs'] / df[f'{col}_len']

            df = df.drop(f'{col}_len', axis = 1)
            df = df.drop(f'match_{col}_len', axis = 1)
            gc.collect()

    return df


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

def analysis(df):
    logger.info('Num of data: %s' % len(df))
    logger.info('Num of unique id: %s' % df['id'].nunique())
    logger.info('Num of unique poi: %s' % df['point_of_interest'].nunique())

    poi_grouped = df.groupby('point_of_interest')['id'].count().reset_index()
    logger.info('Mean num of unique poi: %s' % poi_grouped['id'].mean())


data = pd.read_csv("input/train.csv")

## Data split
kf = GroupKFold(n_splits=2)
for i, (trn_idx, val_idx) in enumerate(kf.split(data, 
                                                data['point_of_interest'], 
                                                data['point_of_interest'])):
    data.loc[val_idx, 'set'] = i

logger.info('Num of train data: %s' % len(data))
logger.info(data['set'].value_counts())

valid_data = data[data['set'] == 0]
train_data = data[data['set'] == 1]

logger.info('Train data: ')
analysis(train_data)
logger.info('Valid data: ')
analysis(valid_data)

train_poi = train_data['point_of_interest'].unique().tolist()
valid_poi = valid_data['point_of_interest'].unique().tolist()

logger.info(set(train_poi) & set(valid_poi))

train_ids = train_data['id'].unique().tolist()
valid_ids = valid_data['id'].unique().tolist()
      
logger.info(set(train_ids) & set(valid_ids))
      
tv_ids_d = {}
tv_ids_d['train_ids'] = train_ids
tv_ids_d['valid_ids'] = valid_ids

#np.save('tv_ids_d.npy', tv_ids_d)

del train_data, valid_data
gc.collect()

data = data.set_index('id')
data = data.loc[tv_ids_d['valid_ids']]
data = data.reset_index()

## Train data generated by knn
id2index_d = dict(zip(data['id'].values, data.index))

tfidf_d = {}
for col in vec_columns:
    tfidf = TfidfVectorizer()
    tv_fit = tfidf.fit_transform(data[col].fillna('nan'))
    tfidf_d[col] = tv_fit

train_data = recall_knn(data, num_neighbors)

data = data.set_index('id')
ids = train_data['id'].tolist()
match_ids = train_data['match_id'].tolist()

poi = data.loc[ids]['point_of_interest'].values
match_poi = data.loc[match_ids]['point_of_interest'].values

train_data['label'] = np.array(poi == match_poi, dtype = np.int8)
del poi, match_poi, ids, match_ids
gc.collect()

logger.info('Num of unique id: %s' % train_data['id'].nunique())
logger.info('Num of train data: %s' % len(train_data))
logger.info('Pos rate: %s' % train_data['label'].mean())
logger.info(train_data.sample(5))


## Eval
data = data.reset_index()

id2poi = get_id2poi(data)
poi2ids = get_poi2ids(data)

eval_df = pd.DataFrame()
eval_df['id'] = data['id'].unique().tolist()
eval_df['match_id'] = eval_df['id']
logger.info('Unique id: %s' % len(eval_df))

eval_df_ = train_data[train_data['label'] == 1][['id', 'match_id']]
eval_df = pd.concat([eval_df, eval_df_])

eval_df = eval_df.groupby('id')['match_id'].\
                        apply(list).reset_index()
eval_df['matches'] = eval_df['match_id'].apply(lambda x: ' '.join(set(x)))
logger.info('Unique id: %s' % len(eval_df))

iou_score = get_score(eval_df)
logger.info('IoU score: %s' % iou_score)


## Add features
count = 0
start_row = 0

data = data.set_index('id')
unique_id = train_data['id'].unique().tolist()
num_split_id = len(unique_id) // num_split
for k in range(1, num_split + 1):
    logger.info('Current split: %s' % k)
    end_row = start_row + num_split_id
    if k < num_split:
        cur_id = unique_id[start_row : end_row]
        cur_data = train_data[train_data['id'].isin(cur_id)]
    else:
        cur_id = unique_id[start_row: ]
        cur_data = train_data[train_data['id'].isin(cur_id)]

    cur_data = add_features(cur_data)
    logger.info(cur_data.shape)
    logger.info(cur_data.sample(1))

    cur_data.to_csv('input/valid_data_candidate_25_faiss_v3_%s.csv' % k, index = False)
    start_row = end_row
    count += len(cur_data)

    del cur_data
    gc.collect()

logger.info(count)

