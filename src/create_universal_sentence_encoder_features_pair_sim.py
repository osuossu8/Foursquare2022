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


import tensorflow_hub as hub

class USEVectorizer:
    def __init__(self):
        self.url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        print('load model ...')
        self.model = hub.load(self.url)

    def get_batches(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_vectors(self, sentences) -> pd.DataFrame:

        batches = list(self.get_batches(sentences, 128))

        print('get vectors ...')
        vectors = []
        for batch in tqdm(batches):
            vec = self.model(batch).numpy().astype(np.float16)
            vectors.append(vec)
        vectors = np.concatenate(vectors, 0)
        return vectors


def cos_sim_matrix(matrix1, matrix2):
    d = matrix1 @ matrix2
    norm1 = (matrix1 * matrix1).sum(axis=1, keepdims=True) ** .5
    norm2 = (matrix2 * matrix2).sum(axis=0, keepdims=True) ** .5
    return d / norm1 / norm2


def get_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


id_2_text_use_vector = unpickle('features/id_2_text_use_vector.pkl')

for path in tqdm([
    'input/train_data1.csv',
    'input/train_data2.csv',
    'input/train_data3.csv',
    'input/train_data4.csv',
    'input/train_data5.csv',
    'input/valid_data1.csv',
    'input/valid_data2.csv',
    'input/valid_data3.csv',
    'input/valid_data4.csv',
    'input/valid_data5.csv'
]):
    path_prefix = path.split('/')[-1].split('.')[0]
    print(path_prefix)

    data = pd.read_csv(path)
    batches = list(get_batches(data, 2048))

    res_sims = []
    for batch in tqdm(batches):
        vec1 = np.stack(batch['id'].map(id_2_text_use_vector))
        vec2 = np.stack(batch['match_id'].map(id_2_text_use_vector)).T
        sims = cos_sim_matrix(vec1, vec2)
        res_sims.append(sims)

    res_sims = np.concatenate(res_sims)
    to_pickle(f'features/sim_text_use_vector_{path_prefix}.pkl', res_sims)

