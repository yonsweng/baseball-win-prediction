'''
Predicting the game of baseball using reinforcement learning
'''
import os
import argparse
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from preprocess import preprocess
from dataset import BaseballDataset
from model import Model


def tensorboard():
    '''
    Draw a graph in the tensorboard
    '''
    dataiter = iter(trainloader)
    start_obs, _, info, _ = dataiter.next()

    static_state = model.get_static_state(
        away_start_bat_ids=info['away_start_bat_ids'].to(device),
        home_start_bat_ids=info['home_start_bat_ids'].to(device),
        away_start_pit_id=info['away_start_pit_id'].to(device),
        home_start_pit_id=info['home_start_pit_id'].to(device),
        away_team_id=info['away_team_id'].to(device),
        home_team_id=info['home_team_id'].to(device)
    )

    dynamic_state = model.get_dynamic_state(
        base1_run_id=start_obs['base1_run_id'].to(device),
        base2_run_id=start_obs['base2_run_id'].to(device),
        base3_run_id=start_obs['base3_run_id'].to(device),
        away_bat_lineup=start_obs['away_bat_lineup'].to(device),
        home_bat_lineup=start_obs['home_bat_lineup'].to(device),
        away_pit_lineup=start_obs['away_pit_lineup'].to(device),
        home_pit_lineup=start_obs['home_pit_lineup'].to(device),
        bat_home_id=start_obs['bat_home_id'].to(device),
        inn_ct=start_obs['inn_ct'].to(device),
        outs_ct=start_obs['outs_ct'].to(device),
        away_score_ct=start_obs['away_score_ct'].to(device),
        home_score_ct=start_obs['home_score_ct'].to(device)
    )

    tb.add_graph(model, [static_state, dynamic_state])


def count_numbers(tmp):
    '''
    Returns:
        num_bats,
        num_pits,
        num_teams
    '''
    bats = list(tmp['BAT_ID']) \
         + list(tmp['BASE1_RUN_ID']) \
         + list(tmp['BASE2_RUN_ID']) \
         + list(tmp['BASE3_RUN_ID'])
    return len(set(bats)), \
           len(tmp['PIT_ID'].unique()), \
           len(tmp['AWAY_TEAM_ID'].unique())


def train():
    '''
    Train the model
    '''
    # Loop over epochs
    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        training_losses = {
            'next_dynamic_state': [],
            'reward': [],
            'done': [],
            'value_away': [],
            'value_home': [],
            'total': []
        }
        validation_losses = {
            'next_dynamic_state': [],
            'reward': [],
            'done': [],
            'value_away': [],
            'value_home': [],
            'total': []
        }

        # Training
        model.train()
        data_size = 0
        # true_positive = 0
        # true_negative = 0
        for start_obs, end_obs, info, targets in trainloader:
            batch_size = start_obs['inn_ct'].shape[0]
            data_size += batch_size

            static_state = model.get_static_state(
                away_start_bat_ids=info['away_start_bat_ids'].to(device),
                home_start_bat_ids=info['home_start_bat_ids'].to(device),
                away_start_pit_id=info['away_start_pit_id'].to(device),
                home_start_pit_id=info['home_start_pit_id'].to(device),
                away_team_id=info['away_team_id'].to(device),
                home_team_id=info['home_team_id'].to(device)
            )

            dynamic_state = model.get_dynamic_state(
                base1_run_id=start_obs['base1_run_id'].to(device),
                base2_run_id=start_obs['base2_run_id'].to(device),
                base3_run_id=start_obs['base3_run_id'].to(device),
                away_bat_lineup=start_obs['away_bat_lineup'].to(device),
                home_bat_lineup=start_obs['home_bat_lineup'].to(device),
                away_pit_lineup=start_obs['away_pit_lineup'].to(device),
                home_pit_lineup=start_obs['home_pit_lineup'].to(device),
                bat_home_id=start_obs['bat_home_id'].to(device),
                inn_ct=start_obs['inn_ct'].to(device),
                outs_ct=start_obs['outs_ct'].to(device),
                away_score_ct=start_obs['away_score_ct'].to(device),
                home_score_ct=start_obs['home_score_ct'].to(device)
            )

            # Forward the model
            next_dynamic_state, reward, done, value_away, value_home = \
                model(static_state, dynamic_state)

            targets['next_dynamic_state'] = model.get_dynamic_state(
                base1_run_id=end_obs['base1_run_id'].to(device),
                base2_run_id=end_obs['base2_run_id'].to(device),
                base3_run_id=end_obs['base3_run_id'].to(device),
                away_bat_lineup=end_obs['away_bat_lineup'].to(device),
                home_bat_lineup=end_obs['home_bat_lineup'].to(device),
                away_pit_lineup=end_obs['away_pit_lineup'].to(device),
                home_pit_lineup=end_obs['home_pit_lineup'].to(device),
                bat_home_id=end_obs['bat_home_id'].to(device),
                inn_ct=end_obs['inn_ct'].to(device),
                outs_ct=end_obs['outs_ct'].to(device),
                away_score_ct=end_obs['away_score_ct'].to(device),
                home_score_ct=end_obs['home_score_ct'].to(device)
            )

            # Get losses
            loss = {}
            loss['next_dynamic_state'] = MSELoss(next_dynamic_state, targets['next_dynamic_state'].to(device))
            loss['reward'] = MSELoss(reward, targets['reward'].to(device))
            loss['done'] = BCELoss(done, targets['done'].to(device))
            loss['value_away'] = MSELoss(value_away, targets['value_away'].to(device))
            loss['value_home'] = MSELoss(value_home, targets['value_home'].to(device))
            loss['total'] = sum(loss.values())

            # Save the losses
            for key in loss:
                training_losses[key].append(loss[key].tolist() * batch_size)

            # Optimize
            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()

        # Print training losses
        for key in training_losses:
            print(f'training {key} loss: {sum(training_losses[key]) / data_size}')

            # For TensorBoard
            tb.add_scalar(f'training {key} loss', sum(training_losses[key]) / data_size, epoch)

        # Validation
        model.eval()
        data_size = 0
        # true_positive = 0
        # true_negative = 0
        for start_obs, end_obs, info, targets in validloader:
            batch_size = start_obs['inn_ct'].shape[0]
            data_size += batch_size

            static_state = model.get_static_state(
                away_start_bat_ids=info['away_start_bat_ids'].to(device),
                home_start_bat_ids=info['home_start_bat_ids'].to(device),
                away_start_pit_id=info['away_start_pit_id'].to(device),
                home_start_pit_id=info['home_start_pit_id'].to(device),
                away_team_id=info['away_team_id'].to(device),
                home_team_id=info['home_team_id'].to(device)
            )

            dynamic_state = model.get_dynamic_state(
                base1_run_id=start_obs['base1_run_id'].to(device),
                base2_run_id=start_obs['base2_run_id'].to(device),
                base3_run_id=start_obs['base3_run_id'].to(device),
                away_bat_lineup=start_obs['away_bat_lineup'].to(device),
                home_bat_lineup=start_obs['home_bat_lineup'].to(device),
                away_pit_lineup=start_obs['away_pit_lineup'].to(device),
                home_pit_lineup=start_obs['home_pit_lineup'].to(device),
                bat_home_id=start_obs['bat_home_id'].to(device),
                inn_ct=start_obs['inn_ct'].to(device),
                outs_ct=start_obs['outs_ct'].to(device),
                away_score_ct=start_obs['away_score_ct'].to(device),
                home_score_ct=start_obs['home_score_ct'].to(device)
            )

            # Forward the model
            next_dynamic_state, reward, done, value_away, value_home = \
                model(static_state, dynamic_state)

            targets['next_dynamic_state'] = model.get_dynamic_state(
                base1_run_id=end_obs['base1_run_id'].to(device),
                base2_run_id=end_obs['base2_run_id'].to(device),
                base3_run_id=end_obs['base3_run_id'].to(device),
                away_bat_lineup=end_obs['away_bat_lineup'].to(device),
                home_bat_lineup=end_obs['home_bat_lineup'].to(device),
                away_pit_lineup=end_obs['away_pit_lineup'].to(device),
                home_pit_lineup=end_obs['home_pit_lineup'].to(device),
                bat_home_id=end_obs['bat_home_id'].to(device),
                inn_ct=end_obs['inn_ct'].to(device),
                outs_ct=end_obs['outs_ct'].to(device),
                away_score_ct=end_obs['away_score_ct'].to(device),
                home_score_ct=end_obs['home_score_ct'].to(device)
            )

            # Get losses
            loss = {}
            loss['next_dynamic_state'] = MSELoss(next_dynamic_state, targets['next_dynamic_state'].to(device))
            loss['reward'] = MSELoss(reward, targets['reward'].to(device))
            loss['done'] = BCELoss(done, targets['done'].to(device))
            loss['value_away'] = MSELoss(value_away, targets['value_away'].to(device))
            loss['value_home'] = MSELoss(value_home, targets['value_home'].to(device))
            loss['total'] = sum(loss.values())

            # Save the losses
            for key in loss:
                validation_losses[key].append(loss[key].tolist() * batch_size)

        # Print validation losses
        for key in validation_losses:
            print(f'validation {key} loss: {sum(validation_losses[key]) / data_size}')

            # For TensorBoard
            tb.add_scalar(f'validation {key} loss', sum(validation_losses[key]) / data_size, epoch)

        # Draw histograms for model weights
        tb.add_histogram('bat_emb', model.bat_emb.weight, epoch)
        tb.add_histogram('pit_emb', model.pit_emb.weight, epoch)
        tb.add_histogram('team_emb', model.team_emb.weight, epoch)

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-6, metavar='R',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--embedding-dim', type=int, default=256, metavar='N',
                        help='embedding dimension (default: 256)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='cuda device number (default: 1)')
    parser.add_argument('--seed', type=int, default=777, metavar='N',
                        help='random seed (default: 777)')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set cuda
    device = torch.device(f'cuda:{args.cuda}')

    # Load data
    data = pd.read_csv('input/mlbplaybyplay2010s/all2010.csv', low_memory=False)

    # Count the numbers of batters, pitchers and teams.
    num_bats, num_pits, num_teams = count_numbers(data)
    print(f'# of batters: {num_bats}, # of pitchers: {num_pits}, # of teams: {num_teams}')

    # Load preprocessed data or make it.
    PREPROCESSED_PATH = 'input/preprocessed/all2010.csv'
    if os.path.exists(PREPROCESSED_PATH):
        data = pd.read_csv(PREPROCESSED_PATH, low_memory=False)
    else:
        data = preprocess(data)
        data.to_csv(PREPROCESSED_PATH, index=False)

    # Train-test split
    games = [game for _, game in data.groupby(data['GAME_ID'])]
    train_games, test_games = train_test_split(games, test_size=0.2)
    train_games, valid_games = train_test_split(train_games, test_size=0.2)
    train_games = pd.concat(train_games, ignore_index=True)
    valid_games = pd.concat(valid_games, ignore_index=True)
    test_games = pd.concat(test_games, ignore_index=True)

    # Dataset and dataloader
    trainset = BaseballDataset(train_games)
    validset = BaseballDataset(valid_games)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=32)
    validloader = torch.utils.data.DataLoader(validset,
        batch_size=args.batch_size, shuffle=True, num_workers=32)

    # Initiate the model
    model = Model(num_bats, num_pits, num_teams, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    MSELoss = torch.nn.MSELoss()
    BCELoss = torch.nn.BCELoss()

    # For TensorBoard
    tb = SummaryWriter()

    tensorboard()
    train()

    tb.close()
