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


def add_sep_token(df):
    # Before concatenation, fill NAN with unknown
    df.fillna('unknown', inplace = True)
    df['text'] = \
      df['near_name_0'] + '[SEP]' + df['near_address_0'] + '[SEP]' + df['near_city_0'] + '[SEP]' \
    + df['near_state_0'] + '[SEP]' + df['near_country_0'] + '[SEP]' + df['near_url_0'] + '[SEP]' + df['near_categories_0'] + '[SEP]' \
    + df['near_name_1'] + '[SEP]' + df['near_address_1'] + '[SEP]' + df['near_city_1'] + '[SEP]' \
    + df['near_state_1'] + '[SEP]' + df['near_country_1'] + '[SEP]' + df['near_url_1'] + '[SEP]' + df['near_categories_1'] + '[SEP]' \
    + df['near_name_2'] + '[SEP]' + df['near_address_2'] + '[SEP]' + df['near_city_2'] + '[SEP]' \
    + df['near_state_2'] + '[SEP]' + df['near_country_2'] + '[SEP]' + df['near_url_2'] + '[SEP]' + df['near_categories_2'] \
    + df['near_name_3'] + '[SEP]' + df['near_address_3'] + '[SEP]' + df['near_city_3'] + '[SEP]' \
    + df['near_state_3'] + '[SEP]' + df['near_country_3'] + '[SEP]' + df['near_url_3'] + '[SEP]' + df['near_categories_3'] + '[SEP]' \
    + df['near_name_4'] + '[SEP]' + df['near_address_4'] + '[SEP]' + df['near_city_4'] + '[SEP]' \
    + df['near_state_4'] + '[SEP]' + df['near_country_4'] + '[SEP]' + df['near_url_4'] + '[SEP]' + df['near_categories_4'] \
    + df['near_name_5'] + '[SEP]' + df['near_address_5'] + '[SEP]' + df['near_city_5'] + '[SEP]' \
    + df['near_state_5'] + '[SEP]' + df['near_country_5'] + '[SEP]' + df['near_url_5'] + '[SEP]' + df['near_categories_5'] + '[SEP]' \
    + df['near_name_6'] + '[SEP]' + df['near_address_6'] + '[SEP]' + df['near_city_6'] + '[SEP]' \
    + df['near_state_6'] + '[SEP]' + df['near_country_6'] + '[SEP]' + df['near_url_6'] + '[SEP]' + df['near_categories_6'] \
    + df['near_name_7'] + '[SEP]' + df['near_address_7'] + '[SEP]' + df['near_city_7'] + '[SEP]' \
    + df['near_state_7'] + '[SEP]' + df['near_country_7'] + '[SEP]' + df['near_url_7'] + '[SEP]' + df['near_categories_7'] + '[SEP]' \
    + df['near_name_8'] + '[SEP]' + df['near_address_8'] + '[SEP]' + df['near_city_8'] + '[SEP]' \
    + df['near_state_8'] + '[SEP]' + df['near_country_8'] + '[SEP]' + df['near_url_8'] + '[SEP]' + df['near_categories_8'] \
    + df['near_name_9'] + '[SEP]' + df['near_address_9'] + '[SEP]' + df['near_city_9'] + '[SEP]' \
    + df['near_state_9'] + '[SEP]' + df['near_country_9'] + '[SEP]' + df['near_url_9'] + '[SEP]' + df['near_categories_9']
    
    for i in range(10):
        del df[f'near_name_{i}'], df[f'near_address_{i}'], df[f'near_city_{i}'], df[f'near_state_{i}'], df[f'near_country_{i}'], df[f'near_url_{i}'], df[f'near_categories_{i}']
        gc.collect()
    return df


class CFG:
    EXP_ID = '008'
    seed = 71
    epochs = 5
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
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = 180


import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


print('load data')
train = pd.read_csv('input/train_with_near_candidate_target_v2.csv')
train['num_target'] = train[[f"target_{i}" for i in range(10)]].sum(1)


print('load features')

train = add_sep_token(train)


kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(train, train["num_target"], train["num_target"]))):
    train.loc[val_idx, "fold"] = i

print(train.shape)
print(train.head())


class FoursquareDataset:
    def __init__(self, text, y, tokenizer=CFG.tokenizer, max_len=CFG.max_len):
        self.text = text
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        inputs = self.tokenizer(
            text,
            add_special_tokens = True,
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True,
            return_offsets_mapping = False,
            return_token_type_ids = False,
            return_attention_mask = False,
        )

        ids = inputs["input_ids"]
        targets = self.y[item]

        return {
            "x": torch.tensor(ids, dtype=torch.long),
            "y" : torch.tensor(targets, dtype=torch.float32),
        }


class FoursquareModel(nn.Module):
    def __init__(self, model_path):
        super(FoursquareModel, self).__init__()
        self.in_features = 768
        self.bert_model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.l0 = nn.Linear(self.in_features, 10)

    def forward(self, ids):
        bert_outputs = self.bert_model(ids)

        x = bert_outputs[0] # bs, 768

        logits = self.l0(self.dropout(x))
        return logits


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


def loss_fn(logits, targets):
    loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
    loss = loss_fct(logits, targets)
    return loss


def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['x'].to(device)
        targets = data['y'].to(device)

        with autocast(enabled=CFG.apex):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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


def run_one_fold(fold, df):
    trn_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)

    train_dataset = FoursquareDataset(text=trn_df['text'].values, y=trn_df[[f"target_{i}" for i in range(10)]].values)
    train_loader = torch.utils.data.DataLoader(
                   train_dataset, shuffle=True,
                   batch_size=CFG.train_bs,
                   num_workers=0, pin_memory=True)

    val_dataset = FoursquareDataset(text=val_df['text'].values, y=val_df[[f"target_{i}" for i in range(10)]].values)
    val_loader = torch.utils.data.DataLoader(
                 val_dataset, shuffle=False,
                 batch_size=CFG.valid_bs,
                 num_workers=0, pin_memory=True)

    del train_dataset, val_dataset; gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoursquareModel(CFG.model_name)
    model = model.to(device)

    optimizer = transformers.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.ETA_MIN, T_max=500)

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


def calc_cv_and_inference(df):
    model_paths = [OUTPUT_DIR+f'fold-{i}.bin' for i in range(CFG.n_splits)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for p in model_paths:
        model = FoursquareModel(CFG.model_name)
        model.to(device)
        model.load_state_dict(torch.load(p))
        model.eval()
        models.append(model)

    y_true = []
    y_pred = []
    idx = []
    for fold, model in enumerate(models):
        val_df = df[df.fold == fold].reset_index(drop=True)
          
        valid_dataset = FoursquareDataset(text=val_df['text'].values, y=val_df[[f"target_{i}" for i in range(10)]].values)
        valid_dataloader = torch.utils.data.DataLoader(
                 valid_dataset, shuffle=False, 
                 batch_size=CFG.valid_bs,
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
    if fold not in CFG.folds:
        continue
    logger.info("Starting fold {} ...".format(fold))

    run_one_fold(fold, train)

print('train finished')


oof = calc_cv_and_inference(train, features)


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


