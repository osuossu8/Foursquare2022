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

embedder_mpnet = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

text_2_text_mpnet_vector = {}
for v in ['name', 'categories', 'address', 'state', 'url']:
    data[v] = data[v].fillna(f'no{v}')
    unique_value = data[v].unique()
    print(unique_value.shape)
    batches = list(get_batches(unique_value, 128))
    vectors = []
    for batch in tqdm(batches):
        vec = embedder_mpnet.encode(batch, show_progress_bar=False).astype(np.float16)
        vectors.append(vec)
    vectors = np.concatenate(vectors, 0)
    print(vectors.shape)
    text_2_text_mpnet_vector[v] = {k:v for k, v in zip(unique_value, vectors)}

print(text_2_text_mpnet_vector.keys())
to_pickle(f'features/text_2_text_mpnet_vector.pkl', text_2_text_mpnet_vector)
