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


data = pd.read_csv('input/train.csv')

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

id_2_w2v_vector_dict = {}
for idx in ['train_ids', 'valid_ids']:
    data2 = data.set_index('id')
    data2 = data2.loc[tv_ids_d[idx]]
    data2 = data2.reset_index()

    for v in tqdm(vec_columns):
        w2v_vec = W2VVectorizer(data2[v].fillna(f'no{v}'), vec_size=50).get_vectors()
        print(w2v_vec.shape)

        id_2_w2v_vector_dict[v] = {k:v for k, v in zip(data2['id'], w2v_vec)}

    print(id_2_w2v_vector_dict.keys())
    to_pickle(f'features/id_2_w2v_vector_dict_50d_{idx}.pkl', id_2_w2v_vector_dict)

