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

train = pd.read_csv('input/train_data1.csv', nrows=100)
print(train.head())

data = pd.read_csv("input/train.csv")
print(data.head())


data['text'] = ''
for v in vec_columns:
    data['text'] += data[v].fillna('nan') + ' '


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


vectors = USEVectorizer().get_vectors(data['text'].values) 
print(vectors.shape)

id_2_text_use_vector = {k:v for k, v in zip(data['id'].values, vectors)}


to_pickle('features/id_2_text_use_vector.pkl', id_2_text_use_vector)

