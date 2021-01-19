import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from .dataset import BaseballDataset
from src.preprocess import *


def init(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')

    time = datetime.now().strftime('%m-%d-%H-%M-%S')
    tag = time + '_' + '_'.join([f'{arg}={getattr(args, arg)}' for arg in vars(args)])
    return tag, device


def load_data():
    ORIGINAL_PATH = './input/mlbplaybyplay2010s/all2010.csv'
    PREPROCESSED_PATH = './input/preprocessed/all2010.csv'
    if os.path.exists(PREPROCESSED_PATH):
        data = pd.read_csv(PREPROCESSED_PATH, low_memory=False)
    else:
        data = pd.read_csv(ORIGINAL_PATH, low_memory=False)
        data = preprocess(data)
        data.to_csv(PREPROCESSED_PATH, index=False)
    return data


def count_numbers(data):
    '''
    Returns:
        num_bats,
        num_pits,
        num_teams
    '''
    bats = list(data['BAT_ID']) \
         + list(data['BASE1_RUN_ID']) \
         + list(data['BASE2_RUN_ID']) \
         + list(data['BASE3_RUN_ID'])
    num_bats = len(set(bats))
    num_pits = len(data['PIT_ID'].unique())
    num_teams = len(data['AWAY_TEAM_ID'].unique())
    print(f'# of batters: {num_bats}, # of pitchers: {num_pits}, # of teams: {num_teams}')
    return num_bats, num_pits, num_teams


def get_dataloaders(data, args):
    # Train-valid-test split
    games = [game for _, game in data.groupby(data['GAME_ID'])]
    train_games, test_games = train_test_split(games, test_size=0.2, random_state=args.seed)
    train_games, valid_games = train_test_split(train_games, test_size=0.1, random_state=args.seed)
    train_games = pd.concat(train_games, ignore_index=True)
    valid_games = pd.concat(valid_games, ignore_index=True)
    test_games = pd.concat(test_games, ignore_index=True)
    test_games = test_games[test_games['GAME_NEW_FL'] == 'T'].reset_index(drop=True)
    tnew_games = train_games[train_games['GAME_NEW_FL'] == 'T'].reset_index(drop=True)
    vnew_games = valid_games[valid_games['GAME_NEW_FL'] == 'T'].reset_index(drop=True)
    # vnew_games = valid_games[valid_games['INN_CT'] >= 7].reset_index(drop=True)  # 7회 이후만

    trainset = BaseballDataset(train_games)
    validset = BaseballDataset(valid_games)
    tnewset = BaseballDataset(tnew_games)
    vnewset = BaseballDataset(vnew_games)
    testset = BaseballDataset(test_games)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validloader = torch.utils.data.DataLoader(validset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    tnewloader = torch.utils.data.DataLoader(tnewset,
        batch_size=1, shuffle=True, num_workers=args.workers)
    vnewloader = torch.utils.data.DataLoader(vnewset,
        batch_size=1, shuffle=False, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=1, shuffle=False, num_workers=args.workers)

    return trainloader, validloader, tnewloader, vnewloader, testloader


def get_latest_file_path(folder):
    files_Path = folder  # 파일들이 들어있는 폴더
    file_name_and_time_lst = []
    for f_name in os.listdir(f"{files_Path}"):
        # if f_name.startswith(prefix):
        written_time = os.path.getctime(os.path.join(files_Path, f_name))
        file_name_and_time_lst.append((f_name, written_time))
    sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)
    recent_file = sorted_file_lst[0]
    recent_file_name = recent_file[0]
    return os.path.join(folder, recent_file_name)


def select_action(state, model):
    bat_dest, run1_dest, run2_dest, run3_dest, value = model(**state)
    bat_dest = F.softmax(bat_dest, dim=1)
    run1_dest = F.softmax(run1_dest, dim=1)
    run2_dest = F.softmax(run2_dest, dim=1)
    run3_dest = F.softmax(run3_dest, dim=1)
    value = value.reshape(-1)

    bat_dest = Categorical(bat_dest.squeeze())
    run1_dest = Categorical(run1_dest.squeeze())
    run2_dest = Categorical(run2_dest.squeeze())
    run3_dest = Categorical(run3_dest.squeeze())
    bat_act = bat_dest.sample()
    run1_act = run1_dest.sample()
    run2_act = run2_dest.sample()
    run3_act = run3_dest.sample()

    model.saved_log_probs.append(
        bat_dest.log_prob(bat_act) +
        run1_dest.log_prob(run1_act) +
        run2_dest.log_prob(run2_act) +
        run3_dest.log_prob(run3_act)
    )
    model.values.append(value)

    return bat_act.item(), run1_act.item(), run2_act.item(), run3_act.item()
