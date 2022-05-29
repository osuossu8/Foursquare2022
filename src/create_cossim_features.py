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


def get_text(df):
    # Before concatenation, fill NAN with unknown
    df.fillna('unknown', inplace = True)
    df['text'] = df['name'] + ' ' + df['address'] + ' ' + df['city'] + ' ' + df['state'] + ' ' + df['country'] + ' ' + df['url'] + ' ' + df['categories'] 
    return df


def cos_sim_matrix(matrix1, matrix2):
    d = matrix1 @ matrix2
    norm1 = (matrix1 * matrix1).sum(axis=1, keepdims=True) ** .5
    norm2 = (matrix2 * matrix2).sum(axis=0, keepdims=True) ** .5
    return d / norm1 / norm2


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

#OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
#if not os.path.exists(OUTPUT_DIR):
#    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
#logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")



train = pd.read_csv('input/train_with_near_candidate_target.csv')
train = train.reset_index(drop=True)

train = get_text(train)

print(train.shape)

max_features = 100000
n_components = 16

model = make_pipeline(
                TfidfVectorizer(stop_words='english',
                            binary=True,
                            max_features=max_features),
                TruncatedSVD(n_components=n_components, random_state=42),
             )

# save memory
text_embeddings = model.fit_transform(train['text']).astype(np.float16)

print(text_embeddings.shape)


id_idx_map = {v: k for k, v in zip(list(train.index), train['id'].values)}


sim_all = []
for r in tqdm(range(len(train))):
    row = train.loc[r]
    id_0 = row['near_id_0']
    vec_0 = text_embeddings[id_idx_map[id_0]]
    sims = []
    for i in range(CFG.n_neighbors):
        if i == 0:
            continue
        id_i = row[f'near_id_{i}']
        if id_i == 'unknown':
            sims.append(0)
        else:
            vec_i = text_embeddings[id_idx_map[id_i]]
            sim_0_i = cos_sim_matrix(vec_0.reshape(1, -1), vec_i.reshape(-1, 1))[0][0]
            sims.append(sim_0_i)
    sim_all.append(sims)

print(len(sim_all))


to_pickle('features/near_cossim_features.pkl', np.array(sim_all))


