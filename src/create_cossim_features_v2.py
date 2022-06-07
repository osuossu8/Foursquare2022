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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle


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


train = pd.read_csv('input/train_with_near_candidate_target_v2.csv')

id2index_d = dict(zip(train['id'].values, train.index))

vec_columns = ['name', 'categories', 'address',
               'state', 'url', 'country']

tfidf_d = {}
for col in vec_columns:
    tfidf = TfidfVectorizer()
    tv_fit = tfidf.fit_transform(train[col].fillna('nan'))
    tfidf_d[col] = tv_fit

new_cols = []
ids = train['id'].values
for col in tqdm(vec_columns):
    for j in tqdm(range(1, CFG.n_neighbors)):
        train[f'{col}_sim_0_{j}'] = np.nan
        tv_fit = tfidf_d[col]
        indexs = []
        match_indexs = []
        for l, i in enumerate(train[f'near_id_{j}'].values):
            if str(i) != 'nan':
                indexs.append(id2index_d[ids[l]])
                match_indexs.append(id2index_d[i])
        train.loc[indexs, f'{col}_sim_0_{j}'] = np.array(tv_fit[indexs].multiply(tv_fit[match_indexs]).sum(axis = 1)).ravel()
        new_cols.append(f'{col}_sim_0_{j}')
    
print(len(new_cols))

train[new_cols] = train[new_cols].astype(np.float16)

to_pickle('features/cat_cossim_features.pkl', train[new_cols])

print(train[new_cols].shape)

