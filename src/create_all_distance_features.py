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


def _add_distance_features(args):
    _, df = args

    columns = ['name', 'address', 'city', 'state',
           'zip', 'country', 'url', 'phone', 'categories', 'id']

    for i in tqdm(range(CFG.n_neighbors)):
        latdiffs = []
        londiffs = []
        manhattans = []
        haversines = []
        for c in columns:
            geshs = []
            levens = []
            jaros = []
            lcss = []
            for str1, str2 in df[[f"near_{c}_0", f"near_{c}_{i}"]].values.astype(str):
                if c == 'id':
                    try:
                        lat1 = id_2_lat[str1]
                        lat2 = id_2_lat[str2]
                        lon1 = id_2_lon[str1]
                        lon2 = id_2_lon[str2]
                    except:
                        lat1 = np.nan
                        lat2 = np.nan
                        lon1 = np.nan
                        lon2 = np.nan

                    latdiffs.append((lat1 - lat2))
                    londiffs.append((lon1 - lon2))
                    manhattans.append(manhattan(lat1, lon1, lat2, lon2))
                    haversines.append(vectorized_haversine(lat1, lat2, lon1, lon2))
                else:
                    if str1==str1 and str2==str2:
                        geshs.append(difflib.SequenceMatcher(None, str1, str2).ratio())
                        levens.append(Levenshtein.distance(str1, str2))
                        jaros.append(Levenshtein.jaro_winkler(str1, str2))
                        lcss.append(LCS(str(str1), str(str2)))
                    else:
                        geshs.append(np.nan)
                        levens.append(np.nan)
                        jaros.append(np.nan)
                        lcss.append(np.nan)

            if c == 'id':
                pass
            else:
                df[f"near_{c}_{i}_gesh"] = geshs
                df[f"near_{c}_{i}_leven"] = levens
                df[f"near_{c}_{i}_jaro"] = jaros
                df[f"near_{c}_{i}_lcs"] = lcss

                if not c in ['country', "phone", "zip"]:
                    df[f"near_{c}_{i}_len"] = df[f"near_{c}_{i}"].astype(str).map(len)
                    df[f"near_{c}_{i}_nleven"] = df[f"near_{c}_{i}_leven"] / df[[f"near_{c}_{i}_len", f"near_{c}_0_len"]].max(axis=1)
                    df[f"near_{c}_{i}_nlcsi"] = df[f"near_{c}_{i}_lcs"] / df[f"near_{c}_{i}_len"]
                    df[f"near_{c}_{i}_nlcs0"] = df[f"near_{c}_{i}_lcs"] / df[f"near_{c}_0_len"]

        df[f'latdiff_0_{i}'] = latdiffs
        df[f'londiff_0_{i}'] = londiffs
        df[f'manhattan_0_{i}'] = manhattans
        df[f'euclidean_0_{i}'] = (df[f'latdiff_0_{i}'] ** 2 + df[f'londiff_0_{i}'] ** 2) ** 0.5
        df[f'haversine_0_{i}'] = haversines
        col_64 = list(df.dtypes[df.dtypes == np.float64].index)
        for col in col_64:
            df[col] = df[col].astype(np.float16)

    # del latdiffs, londiffs, manhattans, geshs, levens, jaros, lcss; gc.collect()
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

for i in ['latdiff_0_', 'londiff_0_', 'manhattan_0_', 'euclidean_0_', 'haversine_0_']:
    features += [c for c in train.columns if i in c]

print(features)

train[features] = train[features].astype(np.float16)

to_pickle('features/all_distaice_features.pkl', train[features])

print(train[features].shape)

