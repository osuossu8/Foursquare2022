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

import transformers


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
    EXP_ID = '007'
    seed = 71
    epochs = 15
    #folds = [0, 1, 2, 3, 4]
    #N_FOLDS = 5
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

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


print('load data')
train = pd.read_csv('input/train_with_near_candidate_target.csv')
train['num_target'] = train[[f"target_{i}" for i in range(10)]].sum(1)


print('load features')

distance_features = unpickle('features/distance_features_v2.pkl')
features = list(distance_features.columns)

lat_lon_distance_features = unpickle('features/lon_lat_distaice_features.pkl')
features += list(lat_lon_distance_features.columns)

train = train[[CFG.target, "num_target", "id"] + [f"target_{i}" for i in range(10)] + [f"near_id_{i}" for i in range(CFG.n_neighbors)]]

train = pd.concat([train, distance_features, lat_lon_distance_features], 1)
train[features] = train[features].astype(np.float16)

train.reset_index(drop=True, inplace=True)

kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(train, train["num_target"], train["num_target"]))):
    train.loc[val_idx, "fold"] = i


print(train[features].shape)

print('scaling')

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(train[features].fillna(-1)))
train[features] = X.copy()

to_pickle(OUTPUT_DIR+f'standard_scaler_{CFG.EXP_ID}.pkl', scaler)
print()


class MLPDataset:
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):

        features = self.X[item]

        if self.y is not None:
            targets = self.y[item]
        
            return {
                'x': torch.tensor(features, dtype=torch.float32),
                'y': torch.tensor(targets, dtype=torch.float32),
            }
          
        else:
            return {
                'x': torch.tensor(features, dtype=torch.float32),
            }  


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.score = f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.5, average="micro")
       
        return {
            "score" : self.score,
        }


class MLP(nn.Module):
    def __init__(self, len_features):
        super(MLP, self).__init__()    

        self.head = nn.Sequential(
            nn.Linear(len_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, features):
        # bs, len_features
        output = self.head(features)
        return output


def loss_fn(logits, targets):
    loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
    loss = loss_fct(logits, targets)
    return loss


def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['x'].to(device)
        targets = data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data['x'].to(device)
            targets = data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def run_one_fold(fold, df, features):
    trn_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)

    train_dataset = MLPDataset(X=trn_df[features].values, y=trn_df[[f"target_{i}" for i in range(10)]].values)
    train_loader = torch.utils.data.DataLoader(
                   train_dataset, shuffle=True,
                   batch_size=256,
                   num_workers=0, pin_memory=True)

    val_dataset = MLPDataset(X=val_df[features].values, y=val_df[[f"target_{i}" for i in range(10)]].values)
    val_loader = torch.utils.data.DataLoader(
                 val_dataset, shuffle=False,
                 batch_size=512,
                 num_workers=0, pin_memory=True)

    del train_dataset, val_dataset
    gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(df[features].shape[1])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    patience = 5
    p = 0
    min_loss = 999
    best_score = -np.inf

    for epoch in range(1, CFG.epochs + 1):

        logger.info("Starting {} epoch...".format(epoch))

        start_time = time.time()

        train_avg, train_loss = train_fn(model, train_loader, device, optimizer, scheduler)
        valid_avg, valid_loss = valid_fn(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch} - train_score:{train_avg['score']:0.5f}  valid_score:{valid_avg['score']:0.5f}")

        if valid_avg['score'] > best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['score']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['score']
            p = 0

        p += 1
        if p > patience:
            logger.info(f'Early Stopping')
            break


def calc_cv_and_inference(df, features):
    model_paths = [OUTPUT_DIR+f'fold-{i}.bin' for i in range(CFG.n_splits)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for p in model_paths:
        model = MLP(df[features].shape[1])
        model.to(device)
        model.load_state_dict(torch.load(p))
        model.eval()
        models.append(model)

    scaler = unpickle(OUTPUT_DIR+f'standard_scaler_{CFG.EXP_ID}.pkl')

    y_true = []
    y_pred = []
    idx = []
    for fold, model in enumerate(models):
        val_df = df[df.fold == fold].reset_index(drop=True)
        X = pd.DataFrame(scaler.fit_transform(val_df[features].fillna(-1))).values
        y=val_df[[f"target_{i}" for i in range(10)]].values
             
        valid_dataset = MLPDataset(X=X, y=y)
        valid_dataloader = torch.utils.data.DataLoader(
                 valid_dataset, shuffle=False, 
                 batch_size=512,
                 num_workers=0, pin_memory=True)
        
        final_output = []
        for b_idx, data in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                inputs = data['x'].to(device)
                targets = data['y'].to(device)
                output = model(inputs)
                output = output.detach().cpu().numpy().tolist()
                final_output.extend(output)
        logger.info(f1_score(y, np.array(final_output) > 0.5, average="micro"))
        y_pred.append(np.array(final_output))
        y_true.append(y)
        idx.append(val_df['id'].values)

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    idx = np.concatenate(idx)

    oof_df = pd.DataFrame()
    oof_df[[f"target_{i}" for i in range(10)]] = y_true
    oof_df[[f"oof_{i}" for i in range(10)]] = y_pred
    oof_df['id'] = idx
    oof_df = oof_df.sort_values('id')
    oof_df.to_csv(OUTPUT_DIR+'oof.csv')

    overall_cv_score = f1_score(y_true, y_pred > 0.5, average="micro")
    logger.info(f'cv score {overall_cv_score}')
    return oof_df


for fold in range(CFG.n_splits):
    logger.info("Starting fold {} ...".format(fold))

    run_one_fold(fold, train, features)

print('train finished')


oof = calc_cv_and_inference(train, features)

# oof = pd.read_csv(OUTPUT_DIR+'oof.csv')

train = pd.merge(train, oof, on='id', how='inner')[[CFG.target, "id"] + [f"oof_{i}" for i in range(10)] + [f"near_id_{i}" for i in range(CFG.n_neighbors)]]

oof = train[[f"oof_{i}" for i in range(10)]] > 0.5

res_df = train[[CFG.target, "id"] +[f"near_id_{i}" for i in range(CFG.n_neighbors)]].copy()
res_df = pd.concat([res_df, oof], 1)

print(res_df.head())

def get_matches(row):
    matches = []
    for i in range(CFG.n_neighbors):
        if (row[f"oof_{i}"] == True) & (row[f"near_id_{i}"] is not np.nan):
            matches.append(row[f"near_id_{i}"])
    return ' '.join(set(matches))

tqdm.pandas()
res_df['matches'] = res_df.progress_apply(lambda row: get_matches(row), axis=1)
res_df['matches'] = res_df['matches']+' '+res_df['id']
res_df['matches'] = res_df['matches'].map(lambda x: ' '.join(set(x.split())))

id2poi = get_id2poi(res_df)
poi2ids = get_poi2ids(res_df)

logger.info(f"IoU: {get_score(res_df):.6f}")


