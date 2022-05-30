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


from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle


# ===============================================================================
# Get manhattan distance
# ===============================================================================
def manhattan(lat1, long1, lat2, long2):
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)

# ===============================================================================
# Get haversine distance
# ===============================================================================
def vectorized_haversine(lats1, lats2, longs1, longs2):
    radius = 6371
    dlat=np.radians(lats2 - lats1)
    dlon=np.radians(longs2 - longs1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats1)) \
        * np.cos(np.radians(lats2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d


class CFG:
    # EXP_ID = '001'
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

set_seed(CFG.seed)
device = set_device()


train = pd.read_csv('input/train_with_near_candidate_target.csv')
#train['num_target'] = train[[f"target_{i}" for i in range(10)]].sum(1)


import Levenshtein
import difflib
from requests import get
import multiprocessing
import joblib


id_2_lat = {k:v for k, v in zip(train['id'].values, train['latitude'].values)}
id_2_lon = {k:v for k, v in zip(train['id'].values, train['longitude'].values)}


def _add_distance_features_2(args):
    _, df = args

    for i in tqdm(range(1, CFG.n_neighbors, 1)):
        latdiffs = []
        londiffs = []
        manhattans = []
        haversines = []
        for str1, str2 in tqdm(df[[f"near_id_0", f"near_id_{i}"]].values.astype(str)):
            try:
                lat1 = id_2_lat[str1] # df.loc[df['id']==str1, 'latitude'].to_numpy()[0]
                lat2 = id_2_lat[str2] # df.loc[df['id']==str2, 'latitude'].to_numpy()[0]
                lon1 = id_2_lon[str1] # df.loc[df['id']==str1, 'longitude'].to_numpy()[0]
                lon2 = id_2_lon[str2] # df.loc[df['id']==str2, 'longitude'].to_numpy()[0]
            except:
                lat1 = -1
                lat2 = -1
                lon1 = -1
                lon2 = -1

            latdiffs.append((lat1 - lat2))
            londiffs.append((lon1 - lon2))
            manhattans.append(manhattan(lat1, lon1, lat2, lon2))
            haversines.append(vectorized_haversine(lat1, lat2, lon1, lon2))

        df[f'latdiff_0_{i}'] = latdiffs
        df[f'londiff_0_{i}'] = londiffs
        df[f'manhattan_0_{i}'] = manhattans
        df[f'euclidean_0_{i}'] = (df[f'latdiff_0_{i}'] ** 2 + df[f'londiff_0_{i}'] ** 2) ** 0.5
        df[f'haversine_0_{i}'] = haversines
        col_64 = list(df.dtypes[df.dtypes == np.float64].index)
        for col in col_64:
            df[col] = df[col].astype(np.float32)
    return df


def add_distance_features_2(df):
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        dfs = pool.imap_unordered(_add_distance_features_2, df.groupby('country'))
        dfs = tqdm(dfs)
        dfs = list(dfs)
    df = pd.concat(dfs)
    return df


train = add_distance_features_2(train)


features = []

for i in ['latdiff_0_', 'londiff_0_', 'manhattan_0_', 'euclidean_0_', 'haversine_0_']:
    features += [c for c in train.columns if i in c]

print(features)

train[features] = train[features].astype(np.float16)

to_pickle('features/lon_lat_distaice_features.pkl', train[features])

print(train[features].shape)

