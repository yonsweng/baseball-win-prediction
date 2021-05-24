import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from BaseballDataset import BaseballDataset
from NNet import Represent, IsDone, Predict


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_train_dataset():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_train.csv',
        low_memory=False)
    return BaseballDataset(data)


def get_valid_dataset():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_valid.csv',
        low_memory=False)
    return BaseballDataset(data)


def get_test_dataset():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_test.csv',
        low_memory=False)
    return BaseballDataset(data)


def random_batch(n, batch_size=1):
    indice = list(range(n))
    random.shuffle(indice)
    for start_idx in range(0, n, batch_size):
        yield indice[start_idx:min(start_idx+batch_size, n)]


def sequential_dataset_batch(data, batch_size=1):
    n = len(data)
    for start_idx in range(0, n, batch_size):
        yield data[range(start_idx, min(start_idx+batch_size, n))]


def sequential_list_batch(data, batch_size=1):
    n = len(data)
    for start_idx in range(0, n, batch_size):
        yield data[start_idx:min(start_idx+batch_size, n)]


def unzip_batch(batch):
    return [{column: batch[column][i] for column in batch}
            for i in range(len(batch[list(batch.keys())[0]]))]


def zip_batch(batch):
    zipped = {}
    for state in batch:
        for key, value in state.items():
            if key not in zipped:
                zipped[key] = []
            zipped[key].append(value)
    for key in zipped:
        zipped[key] = np.array(zipped[key])
    return zipped


def select_action(policy: np.array, state, action_space) -> int:
    valids = action_space.get_valid_moves(state)
    policy = policy * valids
    policy_sum = policy.sum()
    if policy_sum:
        policy /= policy_sum
    return np.random.choice(len(policy), p=policy)


def to_input(state, device='cpu'):
    return {
        'float': torch.tensor([[
            state['INN_CT'] / 9,  # 1/9, 2/9, ..., 1, ...
            (state['BAT_HOME_ID'] - 0.5) * 2,  # -1, 1
            (state['OUTS_CT'] + 1) / 3  # 1/3, 2/3, 1
        ]], dtype=torch.float).to(device),
        'bat': torch.tensor([[
            state['BASE1_RUN_ID'],
            state['BASE2_RUN_ID'],
            state['BASE3_RUN_ID'],
            *[state[f'AWAY_BAT{(i+state["AWAY_BAT_LINEUP_ID"]-1)%9+1}_ID']
              for i in range(9)],
            *[state[f'HOME_BAT{(i+state["HOME_BAT_LINEUP_ID"]-1)%9+1}_ID']
              for i in range(9)]
        ]], dtype=torch.long).to(device),
        'pit': torch.tensor([[
            state['AWAY_PIT_ID'],
            state['HOME_PIT_ID']
        ]], dtype=torch.long).to(device),
        'team': torch.tensor([[
            state['AWAY_TEAM_ID'],
            state['HOME_TEAM_ID']
        ]], dtype=torch.long).to(device)
    }


def to_input_batch(state, device='cpu'):
    away_bat_ids = np.stack([state[f'AWAY_BAT{i}_ID'] for i in range(1, 10)])
    home_bat_ids = np.stack([state[f'HOME_BAT{i}_ID'] for i in range(1, 10)])
    return {
        'float': torch.tensor([
            state['INN_CT'] / 9,  # 1/9, 2/9, ..., 1, ...
            (state['BAT_HOME_ID'] - 0.5) * 2,  # -1, 1
            (state['OUTS_CT'] + 1) / 3  # 1/3, 2/3, 1
        ], dtype=torch.float).transpose(0, 1).to(device),
        'bat': torch.tensor([
            state['BASE1_RUN_ID'],
            state['BASE2_RUN_ID'],
            state['BASE3_RUN_ID'],
            *[away_bat_ids[(state['AWAY_BAT_LINEUP_ID']-1+i) % 9,
              np.arange(away_bat_ids.shape[1])] for i in range(9)],
            *[home_bat_ids[(state['HOME_BAT_LINEUP_ID']-1+i) % 9,
              np.arange(home_bat_ids.shape[1])] for i in range(9)]
        ], dtype=torch.long).transpose(0, 1).to(device),
        'pit': torch.tensor([
            state['AWAY_PIT_ID'],
            state['HOME_PIT_ID']
        ], dtype=torch.long).transpose(0, 1).to(device),
        'team': torch.tensor([
            state['AWAY_TEAM_ID'],
            state['HOME_TEAM_ID']
        ], dtype=torch.long).transpose(0, 1).to(device)
    }


def create_nnet(data, args):
    with open('data_info.json', 'r') as f:
        data_info = json.load(f)

    input = to_input(data[0])
    long_features = \
        input['bat'].shape[1] + input['pit'].shape[1] + input['team'].shape[1]

    nnet = NNet(
        n_batters=data_info['n_batters'],
        n_pitchers=data_info['n_pitchers'],
        n_teams=data_info['n_teams'],
        float_features=input['float'].shape[1],
        long_features=long_features,
        policy_dim=args.n_actions
    ).cuda()
    return nn.DataParallel(nnet)


def create_nnets(data, args):
    with open('data_info.json', 'r') as f:
        data_info = json.load(f)

    features = data[0][0]
    long_features = \
        features['bat'].shape[1] + \
        features['pit'].shape[1] + \
        features['team'].shape[1]

    represent = Represent(
        data_info['n_batters'],
        data_info['n_pitchers'],
        data_info['n_teams'],
        features['float'].shape[1],
        long_features,
        args.represent_size,
        args.embedding_size,
        args.hidden_size,
        args.num_blocks
    )

    is_done = IsDone(
        args.represent_size,
        args.hidden_size,
        args.num_blocks
    )

    predict = Predict(
        args.represent_size,
        args.hidden_size,
        args.num_blocks
    )

    return represent, is_done, predict


def copy_nnet(nnet, nnets):
    for cnet in nnets:
        cnet.load_state_dict(nnet.module.state_dict())
