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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from src.machine_learning_util import set_seed, set_device, init_logger, AverageMeter, to_pickle, unpickle


# https://www.kaggle.com/code/columbia2131/foursquare-iou-metrics
def get_id2poi(input_df: pd.DataFrame) -> dict:
    return dict(zip(input_df['id'], input_df['point_of_interest']))

def get_poi2ids(input_df: pd.DataFrame) -> dict:
    return input_df.groupby('point_of_interest')['id'].apply(set).to_dict()

def get_score(input_df: pd.DataFrame):
    scores = []
    for id_str, matches in zip(input_df['id'].to_numpy(), input_df['matches'].to_numpy()):
        targets = poi2ids[id2poi[id_str]]
        preds = set(matches.split())
        score = len((targets & preds)) / len((targets | preds))
        scores.append(score)
    scores = np.array(scores)
    return scores.mean()


class CFG:
    EXP_ID = '016'
    seed = 71
    epochs = 3
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
    folds = [0, 1, 2]
    apex = True
    #model_name = 'xlm-roberta-base'
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #max_len = 180


import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


print('load data')
train = pd.read_csv('input/train_with_near_candidate_target_v2.csv')

for i in range(CFG.n_neighbors):
    train[f"target_{i}"] = (train[CFG.target]==train[f"near_target_{i}"]).astype(int)

train['num_target'] = train[[f"target_{i}" for i in range(10)]].sum(1)

print('load features')

distance_features = unpickle('features/all_distaice_features.pkl')
features = [c for c in distance_features.columns if '_0_0' not in c]

train = train[[CFG.target, "num_target", "id"] + [f"target_{i}" for i in range(10)] + [f"near_id_{i}" for i in range(CFG.n_neighbors)]]

train = pd.concat([train, distance_features], 1)
train[features] = train[features].astype(np.float16)

train.reset_index(drop=True, inplace=True)

kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(train, train["num_target"], train["num_target"]))):
    train.loc[val_idx, "fold"] = i

print(train[features].shape)


import lightgbm as lgb
from typing import List, Tuple, Optional


class MultiLabelDatasetForLGBM(lgb.Dataset):
    """
    Makeshift Class for storing multi label.

    label: numpy.ndarray (n_example, n_target)
    """

    def __init__(
        self, data, label=None, reference=None, weight=None, group=None, init_score=None, silent=False,
        feature_name='auto', categorical_feature='auto', params=None,  free_raw_data=True
    ):
        """Initialize."""
        if label is not None:
            # # make dummy 1D-array
            dummy_label = np.arange(len(data))

        super(MultiLabelDatasetForLGBM, self).__init__(
            data, dummy_label, reference, weight, group, init_score, silent,
            feature_name, categorical_feature, params, free_raw_data)

        self.mult_label = label

    def get_multi_label(self):
        """Get 2D-array label"""
        return self.mult_label

    def set_multi_label(self, multi_label: np.ndarray):
        """Set 2D-array label"""
        self.mult_label = multi_label
        return self


class MultiMSEForLGBM:
    """Self-made multi-task(?) mse for LightGBM."""
    
    def __init__(self, n_target: int=3) -> None:
        """Initialize."""
        self.name = "my_mmse"
        self.n_target = n_target
    
    def __call__(self, preds: np.ndarray, labels: np.ndarray, weight: Optional[np.ndarray]=None) -> float:
        """Calc loss."""
        loss_by_sample = np.sum((preds - labels) ** 2, axis=1)
        loss = np.average(loss_by_sample, weights=weight)
        
        return loss
    
    def _calc_grad_and_hess(
        self, preds: np.ndarray, labels: np.ndarray, weight: Optional[np.ndarray]=None
    ) -> Tuple[np.ndarray]:
        """Calc Grad and Hess"""
        grad = preds - labels
        hess = np.ones_like(preds)     
        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]

        return grad, hess
    
    def return_loss(self, preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
        """Return Loss for lightgbm"""
        labels = data.get_multi_label()  # <= 改造した Dataset から multi-label を受け取る
        weight = data.get_weight()
        n_example = len(labels)
        
        # # reshape preds: (n_target * n_example,) => (n_target, n_example) =>  (n_example, n_target)
        preds = preds.reshape(self.n_target, n_example).T  # <= preds (1D-array) を 2D-array に直す
        # # calc loss
        loss = self(preds, labels, weight)
        
        return self.name, loss, False
    
    def return_grad_and_hess(self, preds: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray]:
        """Return Grad and Hess for lightgbm"""
        labels = data.get_multi_label()  # <= 改造した Dataset から multi-label を受け取る
        weight = data.get_weight()
        n_example = len(labels)
        
        # # reshape preds: (n_target * n_example,) => (n_target, n_example) =>  (n_example, n_target)
        preds = preds.reshape(self.n_target, n_example).T  # <= preds (1D-array) を 2D-array に直す
        # # calc grad and hess.
        grad, hess =  self._calc_grad_and_hess(preds, labels, weight)

        # # reshape grad, hess: (n_example, n_target) => (n_class, n_target) => (n_target * n_example,) 
        grad = grad.T.reshape(n_example * self.n_target)   # <= 1D-array に戻す
        hess = hess.T.reshape(n_example * self.n_target)   # <= 1D-array に戻す
        
        return grad, hess


def multi_class_accuracy_for_lgbm_altered(
    preds: np.ndarray, data: lgb.Dataset, n_class: int=3,
):
    labels = data.get_multi_label()  # (n_example, n_class)
    weight = data.get_weight()  # (n_example,)
    
    n_example = len(labels)
    # # reshape: (n_example * n_class) => (n_class, n_example) => (n_example, n_class)
    preds = preds.reshape(n_class, n_example).T
    labels_true = labels.argmax(axis=1)
    labels_pred = preds.argmax(axis=1)

    score = np.average(labels_pred == labels_true, weights=weight)
    return "my_macc", score, True


def fit_lgbm(X, y, params=None, es_rounds=20, seed=42, N_SPLITS=5,
             n_class=None, model_dir=None, folds=None):
    models = []
    oof = np.zeros((len(y), n_class), dtype=np.float64)

    for i in tqdm(range(CFG.n_splits)):
        logger.info(f"== fold {i} ==")
        trn_idx = folds!=i
        val_idx = folds==i
        X_train, y_train = X[trn_idx], y[trn_idx]
        X_valid, y_valid = X[val_idx], y[val_idx]

        if model_dir is None:
            my_mmse = MultiMSEForLGBM(n_target=10) 
            lgb_tr = MultiLabelDatasetForLGBM(X_train, y_train)     # <= 改造 Dataset
            lgb_val = MultiLabelDatasetForLGBM(X_valid, y_valid)  # <= 改造 Dataset

            model = lgb.train(
                MODEL_PARAMS_LGB, lgb_tr, **FIT_PARAMS_LGB,
                valid_names=['train', 'valid'], valid_sets=[lgb_tr, lgb_val],
                fobj=my_mmse.return_grad_and_hess,                       # <= gradient と hessian を返す関数
                feval=lambda preds, data: [
                    my_mmse.return_loss(preds, data),                    # <= loss を返す関数
                    multi_class_accuracy_for_lgbm_altered(preds, data, n_class=10),  # <= multi-class accuracy (自作)
                ]
            )

            #model = lgbm.LGBMClassifier(**params)
            #model.fit(
            #    X_train, y_train,
            #    eval_set=[(X_valid, y_valid)],
            #    early_stopping_rounds=es_rounds,
            #    eval_metric='logloss',
    #             verbose=-1)
            #    verbose=50)
        else:
            with open(f'{model_dir}/lgbm_fold{i}.pkl', 'rb') as f:
                model = pickle.load(f)

        pred = model.predict_proba(X_valid)
        oof[val_idx] = pred
        models.append(model)

        file = OUTPUT_DIR+f'lgbm_fold{i}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print()

    # cv = (oof.argmax(axis=-1) == y).mean()
    cv = f1_score(np.array(y), oof > 0.5, average="micro")
    logger.info(f"CV-f1: {cv}")
    return oof, models


params = {
    'objective': "logloss",
    'learning_rate': 0.2,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,

    'max_depth': 7,
    'num_leaves': 35,
    'n_estimators': 1000000,
    "colsample_bytree": 0.9,
}

MODEL_PARAMS_LGB = {
    'num_class': 10,  # <= class 数を指定
    "eta": 0.01,
    "metric": "None",
    "first_metric_only": True,
    "max_depth": -1,
    "seed": 42,
    "num_threads": 4,
    "verbose": -1
}
FIT_PARAMS_LGB = {
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "verbose_eval": 50}

oof, models = fit_lgbm(train[features].values, train[[f"target_{i}" for i in range(10)]].values,
                       params=params, n_class=10,
                       N_SPLITS=CFG.n_splits, folds=train["fold"].values)

#oof = np.load(OUTPUT_DIR+'oof.npy')

print(oof.shape)
np.save(OUTPUT_DIR+'oof.npy', oof)


near_ids = train[[f"near_id_{i}" for i in range(CFG.n_neighbors)]].values

def get_matches(row):
    matches = []
    for i in range(CFG.n_neighbors):
        if (row[f"oof_{i}"] == True) & (str(row[f"near_id_{i}"]) != 'nan'):
            matches.append(row[f"near_id_{i}"])
    return ' '.join(set(matches))


train[[f"oof_{i}" for i in range(10)]] = y_pred > 0.5

train['matches'] = train.progress_apply(lambda row: get_matches(row), axis=1)
train['matches'] = train['matches']+' '+train['id']
train['matches'] = train['matches'].map(lambda x: ' '.join(set(x.split())))


id2poi = get_id2poi(train)
poi2ids = get_poi2ids(train)
train["matches"] = matches
logger.info(f"IoU: {get_score(train):.6f}")

