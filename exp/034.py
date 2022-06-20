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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold
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


def post_process(df):
    id2match = dict(zip(df['id'].values, df['matches'].str.split()))

    for base, match in df[['id', 'matches']].values:
        match = match.split()
        if len(match) == 1:
            continue

        for m in match:
            if base not in id2match[m]:
                id2match[m].append(base)
    df['matches'] = df['id'].map(id2match).map(' '.join)
    return df


def categorical_similarity(A, B):
    if not A or not B:
        return -1

    A = set(str(A).split(", "))
    B = set(str(B).split(", "))

    # Find intersection of two sets
    nominator = A.intersection(B)

    similarity_1 = len(nominator) / len(A)
    similarity_2 = len(nominator) / len(B)

    return max(similarity_1, similarity_2)


class CFG:
    EXP_ID = '034'
    seed = 71
    epochs = 3
    LR = 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    train_bs = 64 #16
    valid_bs = 128 # 32
    EARLY_STOPPING = True
    DEBUG = False # True
    target = "point_of_interest"
    n_neighbors = 10
    n_splits = 3
    apex = True
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = 180
    folds = [0, 1, 2]
               

NUM_NEIGHBOR = 20
SEED = 2022
THRESHOLD = 0.5
NUM_SPLIT = 5
TRAIN_FEATURES = ['kdist',
                'kneighbors',
                'kdist_country',
                'kneighbors_country',
                'name_sim',
                'name_gesh',
                'name_leven',
                'name_jaro',
                'name_lcs',
                'name_len_diff',
                'name_nleven',
                'name_nlcsk',
                'name_nlcs',
                'address_sim',
                'address_gesh',
                'address_leven',
                'address_jaro',
                'address_lcs',
                'address_len_diff',
                'address_nleven',
                'address_nlcsk',
                'address_nlcs',
                'city_gesh',
                'city_leven',
                'city_jaro',
                'city_lcs',
                'city_len_diff',
                'city_nleven',
                'city_nlcsk',
                'city_nlcs',
                'state_sim',
                'state_gesh',
                'state_leven',
                'state_jaro',
                'state_lcs',
                'state_len_diff',
                'state_nleven',
                'state_nlcsk',
                'state_nlcs',
                'zip_gesh',
                'zip_leven',
                'zip_jaro',
                'zip_lcs',
                'url_sim',
                'url_gesh',
                'url_leven',
                'url_jaro',
                'url_lcs',
                'url_len_diff',
                'url_nleven',
                'url_nlcsk',
                'url_nlcs',
                'phone_gesh',
                'phone_leven',
                'phone_jaro',
                'phone_lcs',
                'categories_sim',
                'categories_gesh',
                'categories_leven',
                'categories_jaro',
                'categories_lcs',
                'categories_len_diff',
                'categories_nleven',
                'categories_nlcsk',
                'categories_nlcs',
                'country_sim',
                'country_gesh',
                'country_leven',
                'country_nleven',
                
                #'text_1',
                #'text_2',

                'category_venn',
]


import os

OUTPUT_DIR = f'output/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   

set_seed(CFG.seed)
device = set_device()
logger = init_logger(log_file='log/' + f"{CFG.EXP_ID}.log")


data = pd.read_csv("input/train.csv")

id_2_cat = {k:v for k, v in zip(data['id'].values, data['categories'].fillna('nocategories').values)}

del data;gc.collect()

print('load data')
train = pd.read_csv('input/downsampled_with_oof_027_train_data.csv')
print(train['label'].value_counts())
train = train[train['oof']>0.25].reset_index(drop=True)
print(train['label'].value_counts())

id_2_text = unpickle('features/id_2_text.pkl')

train['text_1'] = train['id'].map(id_2_text)
train['text_2'] = train['match_id'].map(id_2_text)
train['text'] = train['text_1'] + ' [SEP] ' + train['text_2']

train['categories_1'] = train['id'].map(id_2_cat)
train['categories_2'] = train['match_id'].map(id_2_cat)
train["category_venn"] = train[["categories_1", "categories_2"]] \
        .progress_apply(lambda row: categorical_similarity(row.categories_1, row.categories_2),
                        axis=1)

print(train.shape)
print(train['label'].value_counts())
print(train[TRAIN_FEATURES].shape)
# print(train[TRAIN_FEATURES].head())


print(train[['text_1', 'text_2', 'category_venn']].head())


#kf = StratifiedGroupKFold(n_splits=CFG.n_splits)
kf = GroupKFold(n_splits=CFG.n_splits)
for i, (trn_idx, val_idx) in tqdm(enumerate(kf.split(X=train, y=train["label"], groups=train["id"]))):
    train.loc[val_idx, "fold"] = i

for i in range(CFG.n_splits):
    print()
    print(train.query(f'fold == {i}')['label'].value_counts())
    print(train.query(f'fold == {i}')['id'].nunique())
    print()


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(train[TRAIN_FEATURES].fillna(-1)))
train[TRAIN_FEATURES] = X.copy()


class FoursquareDataset:
    def __init__(self, text, y, num_features, tokenizer=CFG.tokenizer, max_len=CFG.max_len):
        self.text = text
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_features = num_features

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
        num_features = self.num_features[item]

        return {
            "x": torch.tensor(ids, dtype=torch.long),
            "y": torch.tensor(targets, dtype=torch.float32),
            "num_features": torch.tensor(num_features, dtype=torch.float32),
        }


class FoursquareModel(nn.Module):
    def __init__(self, model_path, len_features):
        super(FoursquareModel, self).__init__()
        self.in_features = 768
        self.bert_model = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)

        self.head = nn.Sequential(
            nn.Linear(len_features+self.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, ids, num_features):
        bert_outputs = self.bert_model(ids)

        x = bert_outputs[1] # [0] # bs, 768

        x = torch.cat([self.dropout(x), num_features], 1)

        logits = self.head(x)
        return logits.squeeze(-1)


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
        self.score = ((np.array(self.y_pred) > 0.5) == np.array(self.y_true)).mean()

        return {
            "score" : self.score,
        }


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loss_fn(logits, targets):
    # loss_fct = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_fct = RMSELoss()
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
        num_features = data['num_features'].to(device)

        with autocast(enabled=CFG.apex):
            outputs = model(inputs, num_features)
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
            num_features = data['num_features'].to(device)

            outputs = model(inputs, num_features)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def run_one_fold(fold, df, features):
    trn_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)

    train_dataset = FoursquareDataset(text=trn_df['text'].values, y=trn_df['label'].values, num_features=trn_df[features].values)
    train_loader = torch.utils.data.DataLoader(
                   train_dataset, shuffle=True,
                   batch_size=CFG.train_bs,
                   num_workers=0, pin_memory=True)

    val_dataset = FoursquareDataset(text=val_df['text'].values, y=val_df['label'].values, num_features=val_df[features].values)
    val_loader = torch.utils.data.DataLoader(
                 val_dataset, shuffle=False,
                 batch_size=CFG.valid_bs,
                 num_workers=0, pin_memory=True)

    del train_dataset, val_dataset; gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoursquareModel(CFG.model_name, len(features))
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


def calc_cv_and_inference(df, features):
    model_paths = [OUTPUT_DIR+f'fold-{i}.bin' for i in range(CFG.n_splits)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()

    df[features] = pd.DataFrame(scaler.fit_transform(df[features].fillna(-1)))
    y = df['label'].values

    valid_dataset = FoursquareDataset(text=df['text'].values, y=df['label'].values, num_features=df[features].values)
    valid_dataloader = torch.utils.data.DataLoader(
                 valid_dataset, shuffle=False,
                 batch_size=CFG.valid_bs,
                 num_workers=0, pin_memory=True)

    y_pred = []
    for fold, p in enumerate(model_paths):
        model = FoursquareModel(CFG.model_name, len(features))
        model.to(device)
        model.load_state_dict(torch.load(p))
        model.eval()

        final_output = []
        for b_idx, data in tqdm(enumerate(valid_dataloader)):
            with torch.no_grad():
                inputs = data['x'].to(device)
                targets = data['y'].to(device)
                num_features = data['num_features'].to(device)
                output = model(inputs, num_features)
                output = output.detach().cpu().numpy().tolist()
                final_output.extend(output)
        logger.info(((np.array(final_output) > 0.5) == np.array(y)).mean())
        y_pred.append(np.array(final_output))

    y_pred = np.mean(y_pred, 0)
    y_true = df['label'].values
    idx = df['id'].values

    #oof_df = pd.DataFrame()
    #oof_df[[f"target_{i}" for i in range(10)]] = y_true
    #oof_df[[f"oof_{i}" for i in range(10)]] = y_pred
    #oof_df['id'] = idx
    #oof_df = oof_df.sort_values('id')
    #oof_df.to_csv(OUTPUT_DIR+'oof.csv')

    overall_cv_score = ((y_pred > 0.5) == y_true).mean()
    logger.info(f'cv score {overall_cv_score}')

    df['pred'] = y_pred > 0.5
    return df


for fold in range(CFG.n_splits):
    if fold not in CFG.folds:
        continue
    logger.info("Starting fold {} ...".format(fold))
    run_one_fold(fold, train, TRAIN_FEATURES)
print('train finished')


id_2_text = unpickle('features/id_2_text.pkl')

test1 = pd.read_csv('input/valid_data1.csv')
test2 = pd.read_csv('input/valid_data2.csv')
test3 = pd.read_csv('input/valid_data3.csv')
test4 = pd.read_csv('input/valid_data4.csv')
test5 = pd.read_csv('input/valid_data5.csv')

test = pd.concat([
    test1, test2, test3, test4, test5
], 0).reset_index(drop=True)

del test1, test2, test3, test4, test5; gc.collect()

test['text_1'] = test['id'].map(id_2_text)
test['text_2'] = test['match_id'].map(id_2_text)
test['text'] = test['text_1'] + ' [SEP] ' + test['text_2']

test['categories_1'] = test['id'].map(id_2_cat)
test['categories_2'] = test['match_id'].map(id_2_cat)
test["category_venn"] = test[["categories_1", "categories_2"]] \
        .progress_apply(lambda row: categorical_similarity(row.categories_1, row.categories_2),
                        axis=1)

test = calc_cv_and_inference(test, TRAIN_FEATURES)

print(test[['id', 'match_id', 'pred']])
test = test[test['pred'] > 0.5][['id', 'match_id']]
print(test['id'].nunique())

test = test.groupby('id')['match_id'].apply(list).reset_index()
print(test.head())

test['matches'] = test['match_id'].apply(lambda x: ' '.join(set(x)))
test['matches'] = test['matches']+' '+test['id']
test['matches'] = test['matches'].map(lambda x: ' '.join(set(x.split())))

train = pd.read_csv('input/train.csv', usecols=['id', 'point_of_interest'])
test = pd.merge(test, train, on='id', how='inner')

del train; gc.collect()

test = post_process(test)

print(test[['id', 'matches']].head(10))

id2poi = get_id2poi(test)
poi2ids = get_poi2ids(test)

logger.info(f"IoU: {get_score(test):.6f}")
