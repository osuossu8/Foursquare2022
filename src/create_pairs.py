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


class CFG:
    seed = 46
    target = "point_of_interest"
    n_neighbors = 10
    n_splits = 3


random.seed(CFG.seed)
os.environ["PYTHONHASHSEED"] = str(CFG.seed)
np.random.seed(CFG.seed)


train = pd.read_csv('input/train_with_near_candidate_target_v2.csv')

for i in range(CFG.n_neighbors):
    del train[f"target_{i}"]; gc.collect()


print(train.shape)


df = []
for i in tqdm(range(1, 10)):
    tmp = train[[c for c in train.columns if '0' in c]+[c for c in train.columns if f'{i}' in c]]
    tmp.columns = [c for c in train.columns if '0' in c]+[c for c in train.columns if '1' in c]
    df.append(tmp)
    del tmp; gc.collect()

df = pd.concat(df, 0).reset_index(drop=True)

df['target'] = (df['near_target_0'] == df['near_target_1']).astype(int)

print(df.shape)
print(df['target'].value_counts())


