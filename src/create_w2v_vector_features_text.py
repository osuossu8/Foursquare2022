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


from gensim.models import word2vec
from gensim.models import KeyedVectors


class W2VVectorizer:
    def __init__(self, sentences, vec_size):
        self.sentences = sentences.apply(lambda x: x.split())
        self.vec_size = vec_size

        print('fit models ...')
        self.model = word2vec.Word2Vec(self.sentences, 
                                       size=self.vec_size,
                                       min_count=1,
                                       window=1,
                                       iter=100)
        
    def get_batches(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def vectorize(self, word_list : list) -> np.array:
        V = []
        for word in word_list:
            try:
                vector = self.model[word]
            except:
                vector = [j for j in range(self.size)]
            V.append(vector)
        return np.mean(V, 0)        

    def get_vectors(self) -> pd.DataFrame:

        batches = list(self.get_batches(self.sentences, 128))

        print('get vectors ...')
        vectors = []
        for batch in tqdm(batches):
            vec = self.sentences.progress_apply(lambda x: self.vectorize(x))
            #vec = pd.DataFrame(np.vstack([x for x in vec]))
            vectors.append(vec)
        #vectors = pd.concat(vectors, 0).reset_index(drop=True)
        vectors = np.concatenate(vectors, 0)
        return vectors


data = pd.read_csv('input/train.csv')
data['text'] = ''
for v in vec_columns:
    data['text'] += data[v].fillna('nan') + ' '

w2v_vec = W2VVectorizer(data['text'], vec_size=50).get_vectors()
print(w2v_vec.shape)

id_2_text_w2v_vector = {k:v for k, v in zip(data['id'], w2v_vec)}

to_pickle(f'features/id_2_text_w2v_vector_50d.pkl', id_2_text_w2v_vector)

