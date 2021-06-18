import json
import random
import numpy as np
import pandas as pd
import torch
from BaseballDataset import BaseballDataset
from NNet import Predict


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_train_dataset(fold):
    data = pd.read_csv(
        f'input/mlbplaybyplay2010s_preprocessed/all2010_train_{fold}.csv',
        low_memory=False)
    return BaseballDataset(data)


def get_train_new_dataset():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_train.csv',
        low_memory=False)
    data = data[data['GAME_NEW_FL'] == 'T']
    return BaseballDataset(data)


def get_valid_dataset(fold):
    data = pd.read_csv(
        f'input/mlbplaybyplay2010s_preprocessed/all2010_valid_{fold}.csv',
        low_memory=False)
    return BaseballDataset(data)


def get_test_dataset():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_test.csv',
        low_memory=False)
    return BaseballDataset(data)


def count_bats_pits_teams():
    with open('data_info.json', 'r') as f:
        data_info = json.load(f)

    return \
        data_info['n_batters'], \
        data_info['n_pitchers'], \
        data_info['n_teams']


def count_events():
    with open('data_info.json', 'r') as f:
        data_info = json.load(f)

    return data_info['events']


def create_nnets(data, args):
    with open('data_info.json', 'r') as f:
        data_info = json.load(f)

    features = data[0][0]
    long_features = \
        features['bat'].shape[1] + \
        features['pit'].shape[1] + \
        features['team'].shape[1]

    predict = Predict(
        args.embedding_size,
        data_info['n_batters'],
        data_info['n_pitchers'],
        data_info['n_teams'],
        features['float'].shape[1],
        long_features,
        args.hidden_size,
        args.num_blocks,
        args.num_linears
    )

    return predict


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def freeze(models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = False


def unfreeze(models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = True
