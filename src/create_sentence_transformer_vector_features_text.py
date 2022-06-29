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
from sentence_transformers import SentenceTransformer

# https://www.kaggle.com/code/guoyonfan/training-data-for-binary-lgb-baseline-0-834/notebook

## Parameters
is_debug = False
SEED = 2022
num_neighbors = 20
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


def get_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


data = pd.read_csv('input/train.csv')
data['text'] = ''
for v in vec_columns:
    data['text'] += data[v].fillna('nan') + ' '

embedder_mpnet = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

## Data split
kf = GroupKFold(n_splits=2)
for i, (trn_idx, val_idx) in enumerate(kf.split(data, 
                                                data['point_of_interest'], 
                                                data['point_of_interest'])):
    data.loc[val_idx, 'set'] = i

valid_data = data[data['set'] == 0]
train_data = data[data['set'] == 1]

train_ids = train_data['id'].unique().tolist()
valid_ids = valid_data['id'].unique().tolist()

tv_ids_d = {}
tv_ids_d['train_ids'] = train_ids
tv_ids_d['valid_ids'] = valid_ids

for idx in ['train_ids', 'valid_ids']:
    data2 = data.set_index('id')
    data2 = data2.loc[tv_ids_d[idx]]
    data2 = data2.reset_index()

    batches = list(get_batches(data2['text'], 128))
    vectors = []
    for batch in tqdm(batches):
        vec = embedder_mpnet.encode(batch, show_progress_bar=False).astype(np.float16)
        vectors.append(vec)
    vectors = np.concatenate(vectors, 0)
    print(vectors.shape)

    id_2_text_mpnet_vector = {k:v for k, v in zip(data2['id'], vectors)}

    to_pickle(f'features/id_2_text_mpnet_vector_{idx}.pkl', id_2_text_mpnet_vector)

