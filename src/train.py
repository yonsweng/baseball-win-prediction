import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from dataset import BaseballDataset
from model import Model
from utils import count_numbers, get_latest_file_path, get_next_bats


def train_dynamics():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.l2)

    MSELoss = torch.nn.MSELoss()
    CELoss = torch.nn.CrossEntropyLoss()

    lr_lambda = lambda x: x / args.warmup if x <= args.warmup else (x / args.warmup) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    clip = lambda x: x.where(x <= 4, torch.tensor([4], dtype=torch.long))

    best_loss = 9.9

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        # Training
        model.train()
        sum_loss = 0.
        for features, targets in trainloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            event_runs_ct, \
            event_outs_ct, \
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest = model.dynamics(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device)
            )
            loss = \
                MSELoss(event_runs_ct, targets['event_runs_ct'].to(device)) + \
                MSELoss(event_outs_ct, targets['event_outs_ct'].to(device)) + \
                CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device))
            sum_loss += event_runs_ct.shape[0] * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        tb.add_scalar('train loss', sum_loss / len(trainloader.dataset), epoch)

        # Validation
        model.eval()
        sum_loss = 0.
        for features, targets in validloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            event_runs_ct, \
            event_outs_ct, \
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest = model.dynamics(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device)
            )
            loss = \
                MSELoss(event_runs_ct, targets['event_runs_ct'].to(device)) + \
                MSELoss(event_outs_ct, targets['event_outs_ct'].to(device)) + \
                CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device))
            sum_loss += event_runs_ct.shape[0] * loss.item()
        tb.add_scalar('valid loss', sum_loss / len(validloader.dataset), epoch)

        # Save the best model.
        if sum_loss / len(validloader.dataset) < best_loss:
            best_loss = sum_loss / len(validloader.dataset)
            torch.save(model.state_dict(), f'../models/dynamics_{tag}.pt')
            print('model saved.')


def train_prediction():
    pretrained_model_path = get_latest_file_path('../models', 'dynamics')
    model.load_state_dict(torch.load(pretrained_model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=args.l2)

    MSELoss = torch.nn.MSELoss()
    CELoss = torch.nn.CrossEntropyLoss()

    lr_lambda = lambda x: x / args.warmup if x <= args.warmup else (x / args.warmup) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    clip = lambda x: x.where(x <= 4, torch.tensor([4], dtype=torch.long))

    best_loss = 99.9

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        # Training
        model.train()
        sum_loss = 0.
        true_negative = 0
        true_positive = 0
        for features, targets in trainloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            away_next_bats_ids = get_next_bats(
                features['away_start_bat_ids'], features['away_bat_lineup'])
            home_next_bats_ids = get_next_bats(
                features['home_start_bat_ids'], features['home_bat_lineup'])
            value_away, value_home = model.prediction(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device),
                away_next_bats_ids.to(device),
                home_next_bats_ids.to(device),
                features['away_start_pit_id'].to(device),
                features['home_start_pit_id'].to(device),
                features['away_team_id'].to(device),
                features['home_team_id'].to(device)
            )
            loss = \
                MSELoss(value_away, targets['value_away'].to(device)) + \
                MSELoss(value_home, targets['value_home'].to(device))
            sum_loss += value_away.shape[0] * loss.item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        tb.add_scalar('train loss', sum_loss / len(trainloader.dataset), epoch)
        tb.add_scalar('train TN', true_negative / len(trainloader.dataset), epoch)
        tb.add_scalar('train TP', true_positive / len(trainloader.dataset), epoch)
        tb.add_scalar('train acc.', (true_negative + true_positive) / len(trainloader.dataset), epoch)

        # Validation
        model.eval()
        sum_loss = 0.
        true_negative = 0
        true_positive = 0
        for features, targets in validloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            away_next_bats_ids = get_next_bats(
                features['away_start_bat_ids'], features['away_bat_lineup'])
            home_next_bats_ids = get_next_bats(
                features['home_start_bat_ids'], features['home_bat_lineup'])
            value_away, value_home = model.prediction(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device),
                away_next_bats_ids.to(device),
                home_next_bats_ids.to(device),
                features['away_start_pit_id'].to(device),
                features['home_start_pit_id'].to(device),
                features['away_team_id'].to(device),
                features['home_team_id'].to(device)
            )
            loss = \
                MSELoss(value_away, targets['value_away'].to(device)) + \
                MSELoss(value_home, targets['value_home'].to(device))
            sum_loss += value_away.shape[0] * loss.item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
        tb.add_scalar('valid loss', sum_loss / len(validloader.dataset), epoch)
        tb.add_scalar('valid TN', true_negative / len(validloader.dataset), epoch)
        tb.add_scalar('valid TP', true_positive / len(validloader.dataset), epoch)
        tb.add_scalar('valid acc.', (true_negative + true_positive) / len(validloader.dataset), epoch)

        true_negative = 0
        true_positive = 0
        for features, targets in vnewloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            away_next_bats_ids = get_next_bats(
                features['away_start_bat_ids'], features['away_bat_lineup'])
            home_next_bats_ids = get_next_bats(
                features['home_start_bat_ids'], features['home_bat_lineup'])
            value_away, value_home = model.prediction(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device),
                away_next_bats_ids.to(device),
                home_next_bats_ids.to(device),
                features['away_start_pit_id'].to(device),
                features['home_start_pit_id'].to(device),
                features['away_team_id'].to(device),
                features['home_team_id'].to(device)
            )
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
        tb.add_scalar('vnew TN', true_negative / len(vnewloader.dataset), epoch)
        tb.add_scalar('vnew TP', true_positive / len(vnewloader.dataset), epoch)
        tb.add_scalar('vnew acc.', (true_negative + true_positive) / len(vnewloader.dataset), epoch)

        # Save the best model.
        if sum_loss / len(validloader.dataset) < best_loss:
            best_loss = sum_loss / len(validloader.dataset)
            torch.save(model.state_dict(), f'../models/prediction_{tag}.pt')
            print('model saved.')


def train():
    pretrained_model_path = get_latest_file_path('../models', 'prediction')
    model.load_state_dict(torch.load(pretrained_model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    MSELoss = torch.nn.MSELoss()
    CELoss = torch.nn.CrossEntropyLoss()

    lr_lambda = lambda x: x / args.warmup if x <= args.warmup else (x / args.warmup) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    clip = lambda x: x.where(x <= 4, torch.tensor([4], dtype=torch.long))

    best_loss = 99.9

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        # Training
        model.train()
        sum_loss = 0.
        true_negative = 0
        true_positive = 0
        for features, targets in trainloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            away_next_bats_ids = get_next_bats(
                features['away_start_bat_ids'], features['away_bat_lineup'])
            home_next_bats_ids = get_next_bats(
                features['home_start_bat_ids'], features['home_bat_lineup'])
            event_runs_ct, \
            event_outs_ct, \
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest, \
            value_away, \
            value_home = model(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device),
                away_next_bats_ids.to(device),
                home_next_bats_ids.to(device),
                features['away_start_pit_id'].to(device),
                features['home_start_pit_id'].to(device),
                features['away_team_id'].to(device),
                features['home_team_id'].to(device)
            )
            loss = \
                MSELoss(event_runs_ct, targets['event_runs_ct'].to(device)) + \
                MSELoss(event_outs_ct, targets['event_outs_ct'].to(device)) + \
                CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device)) + \
                MSELoss(value_away, targets['value_away'].to(device)) + \
                MSELoss(value_home, targets['value_home'].to(device))
            sum_loss += event_runs_ct.shape[0] * loss.item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        tb.add_scalar('train loss', sum_loss / len(trainloader.dataset), epoch)
        tb.add_scalar('train TN', true_negative / len(trainloader.dataset), epoch)
        tb.add_scalar('train TP', true_positive / len(trainloader.dataset), epoch)
        tb.add_scalar('train acc.', (true_negative + true_positive) / len(trainloader.dataset), epoch)

        # Validation
        model.eval()
        sum_loss = 0.
        true_negative = 0
        true_positive = 0
        for features, targets in validloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            away_next_bats_ids = get_next_bats(
                features['away_start_bat_ids'], features['away_bat_lineup'])
            home_next_bats_ids = get_next_bats(
                features['home_start_bat_ids'], features['home_bat_lineup'])
            event_runs_ct, \
            event_outs_ct, \
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest, \
            value_away, \
            value_home = model(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device),
                away_next_bats_ids.to(device),
                home_next_bats_ids.to(device),
                features['away_start_pit_id'].to(device),
                features['home_start_pit_id'].to(device),
                features['away_team_id'].to(device),
                features['home_team_id'].to(device)
            )
            loss = \
                MSELoss(event_runs_ct, targets['event_runs_ct'].to(device)) + \
                MSELoss(event_outs_ct, targets['event_outs_ct'].to(device)) + \
                CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device)) + \
                MSELoss(value_away, targets['value_away'].to(device)) + \
                MSELoss(value_home, targets['value_home'].to(device))
            sum_loss += event_runs_ct.shape[0] * loss.item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
        tb.add_scalar('valid loss', sum_loss / len(validloader.dataset), epoch)
        tb.add_scalar('valid TN', true_negative / len(validloader.dataset), epoch)
        tb.add_scalar('valid TP', true_positive / len(validloader.dataset), epoch)
        tb.add_scalar('valid acc.', (true_negative + true_positive) / len(validloader.dataset), epoch)

        true_negative = 0
        true_positive = 0
        for features, targets in vnewloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            away_next_bats_ids = get_next_bats(
                features['away_start_bat_ids'], features['away_bat_lineup'])
            home_next_bats_ids = get_next_bats(
                features['home_start_bat_ids'], features['home_bat_lineup'])
            event_runs_ct, \
            event_outs_ct, \
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest, \
            value_away, \
            value_home = model(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device),
                away_next_bats_ids.to(device),
                home_next_bats_ids.to(device),
                features['away_start_pit_id'].to(device),
                features['home_start_pit_id'].to(device),
                features['away_team_id'].to(device),
                features['home_team_id'].to(device)
            )
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
        tb.add_scalar('vnew TN', true_negative / len(vnewloader.dataset), epoch)
        tb.add_scalar('vnew TP', true_positive / len(vnewloader.dataset), epoch)
        tb.add_scalar('vnew acc.', (true_negative + true_positive) / len(vnewloader.dataset), epoch)

        # Save the best model.
        if sum_loss / len(validloader.dataset) < best_loss:
            best_loss = sum_loss / len(validloader.dataset)
            torch.save(model.state_dict(), f'../models/trained_{tag}.pt')
            print('model saved.')


def td_zero():
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 자주 바뀌는 순.
    parser.add_argument('--dynamics', action='store_true')
    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--rl', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='F')
    parser.add_argument('--l2', type=float, default=1e-3, metavar='F')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='F')
    parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
    parser.add_argument('--warmup', type=int, default=2000, metavar='N')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N')
    parser.add_argument('--epochs', type=int, default=20, metavar='N')
    parser.add_argument('--seed', type=int, default=777, metavar='N')
    parser.add_argument('--cuda', type=int, default=0, metavar='N')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')

    # Load data
    PREPROCESSED_PATH = '../input/preprocessed/all2010.csv'
    if os.path.exists(PREPROCESSED_PATH):
        data = pd.read_csv(PREPROCESSED_PATH, low_memory=False)
    else:
        data = pd.read_csv('../input/mlbplaybyplay2010s/all2010.csv', low_memory=False)
        data = preprocess(data)
        data.to_csv(PREPROCESSED_PATH, index=False)

    num_bats, num_pits, num_teams = count_numbers(data)
    print(f'# of batters: {num_bats}, # of pitchers: {num_pits}, # of teams: {num_teams}')

    # Train-valid-test split
    games = [game for _, game in data.groupby(data['GAME_ID'])]
    train_games, test_games = train_test_split(games, test_size=0.2, random_state=args.seed)
    train_games, valid_games = train_test_split(train_games, test_size=0.2, random_state=args.seed)
    train_games = pd.concat(train_games, ignore_index=True)
    valid_games = pd.concat(valid_games, ignore_index=True)
    test_games = pd.concat(test_games, ignore_index=True)
    vnew_games = valid_games[valid_games['GAME_NEW_FL'] == 'T'].reset_index(drop=True)

    # Dataset and dataloader
    trainset = BaseballDataset(train_games)
    validset = BaseballDataset(valid_games)
    vnewset = BaseballDataset(vnew_games)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=32)
    validloader = torch.utils.data.DataLoader(validset,
        batch_size=args.batch_size, shuffle=False, num_workers=32)
    vnewloader = torch.utils.data.DataLoader(vnewset,
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = Model(num_bats, num_pits, num_teams, args.emb_dim, args.dropout).to(device)

    time = datetime.now().strftime('%m-%d-%H-%M-%S')
    tag = time + '_' + '_'.join([f'{arg}={getattr(args, arg)}' for arg in vars(args)])

    if args.dynamics:
        tb = SummaryWriter(f'../runs/{"dynamics_" + tag}')
        train_dynamics()
        tb.close()

    if args.prediction:
        tb = SummaryWriter(f'../runs/{"prediction_" + tag}')
        train_prediction()
        tb.close()

    if not args.no_train:
        tb = SummaryWriter(f'../runs/{"train_" + tag}')
        train()
        tb.close()

    if not args.rl:
        tb = SummaryWriter(f'../runs/{"rl_" + tag}')
        td_zero()
        tb.close()
