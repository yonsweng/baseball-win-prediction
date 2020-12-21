'''
Predicting the game of baseball using reinforcement learning
'''
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from preprocess import preprocess
from dataset import BaseballDataset
from model import Model
from embedding import train_embeddings, load_embeddings


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


def tensorboard():
    '''
    Draw a graph of the model on TensorBoard
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


def get_losses(start_obs, end_obs, info, targets):
    '''
    Returns:
        loss (dict)
    '''
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
    dynamic_state = F.normalize(dynamic_state)

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
    targets['next_dynamic_state'] = F.normalize(targets['next_dynamic_state'])

    loss = {}
    loss['next_dynamic_state'] = args.state_loss * MSELoss(next_dynamic_state, targets['next_dynamic_state'].to(device))
    loss['reward'] = MSELoss(reward, targets['reward'].to(device))
    loss['done'] = BCELoss(done, targets['done'].to(device))
    loss['value_away'] = args.value_loss * MSELoss(value_away, targets['value_away'].to(device))
    loss['value_home'] = args.value_loss * MSELoss(value_home, targets['value_home'].to(device))
    loss['total'] = sum(loss.values())

    return loss


def train():
    '''
    Train the model
    '''
    losses = [
        'next_dynamic_state',
        'reward',
        'done',
        'value_away',
        'value_home',
        'total'
    ]
    min_val_total_loss = 99.99
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    MSELoss = torch.nn.MSELoss()
    BCELoss = torch.nn.BCELoss()

    lr_lambda = lambda x: x / args.warmup if x <= args.warmup else (x / args.warmup) ** -0.5
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Loop over epochs
    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        training_losses = {loss: [] for loss in losses}
        validation_losses = {loss: [] for loss in losses}

        # Training
        model.train()
        for start_obs, end_obs, info, targets in trainloader:
            batch_size = start_obs['inn_ct'].shape[0]

            loss = get_losses(start_obs, end_obs, info, targets)

            # Save losses
            for key in loss:
                training_losses[key].append(loss[key].tolist() * batch_size)

            # Optimize
            optimizer.zero_grad()
            loss['total'].backward()
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        for start_obs, end_obs, info, targets in validloader:
            batch_size = start_obs['inn_ct'].shape[0]

            loss = get_losses(start_obs, end_obs, info, targets)

            # Save the losses
            for key in loss:
                validation_losses[key].append(loss[key].tolist() * batch_size)

        # Simulation validation
        true_positive = 0
        true_negative = 0
        for start_obs, end_obs, info, targets in valid_simul_loader:
            batch_size = start_obs['inn_ct'].shape[0]

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
            dynamic_state = F.normalize(dynamic_state)
            
            tb.add_histogram('dynamic_state', dynamic_state, epoch)

            # Forward the model
            _, _, _, value_away, value_home = \
                model(static_state, dynamic_state)

            true_positive += torch.logical_and(
                targets['value_away'] < targets['value_home'],
                value_away.cpu() < value_home.cpu()
            ).squeeze().sum().tolist()

            true_negative += torch.logical_and(
                targets['value_away'] > targets['value_home'],
                value_away.cpu() > value_home.cpu()
            ).squeeze().sum().tolist()

        tb.add_scalar('v(s0) true positive ratio', true_positive / len(valid_simul_loader.dataset), epoch)
        tb.add_scalar('v(s0) true negative ratio', true_negative / len(valid_simul_loader.dataset), epoch)
        tb.add_scalar('v(s0) accuracy', (true_positive + true_negative) / len(valid_simul_loader.dataset), epoch)

        # Save model
        val_total_loss = sum(validation_losses['total']) / len(validloader.dataset)
        if val_total_loss < min_val_total_loss:
            torch.save(model.state_dict(), f'models/model{comment}.pt')
            min_val_total_loss = val_total_loss
            print(f'model{comment}.pt saved')

        # Print the losses
        for loss in losses:
            training_loss = sum(training_losses[loss]) / len(trainloader.dataset)
            validation_loss = sum(validation_losses[loss]) / len(validloader.dataset)

            print(f'training {loss} loss: {training_loss}')
            print(f'validation {loss} loss: {validation_loss}')

            tb.add_scalar(f'training {loss} loss', training_loss, epoch)
            tb.add_scalar(f'validation {loss} loss', validation_loss, epoch)

        # Make histories on TensorBoard
        tb.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        tb.add_histogram('bat_emb', model.bat_emb.weight, epoch)
        tb.add_histogram('pit_emb', model.pit_emb.weight, epoch)
        tb.add_histogram('team_emb', model.team_emb.weight, epoch)


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-emb', action='store_true')
    parser.add_argument('--load-emb', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='F',
                        help='learning rate for training (default: 1e-3)')
    parser.add_argument('--warmup', type=int, default=2000, metavar='N',
                        help='learning rate warmup step (default: 2000)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='F',
                        help='L2 regularization (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='F',
                        help='dropout rate (default: 0.0)')
    parser.add_argument('--emb-dim', type=int, default=32, metavar='N',
                        help='embedding dimension (default: 32)')
    parser.add_argument('--state-loss', type=float, default=1, metavar='F',
                        help='state loss multiple (default: 1)')
    parser.add_argument('--value-loss', type=float, default=1, metavar='F',
                        help='value loss reduction (default: 1)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--cuda', type=int, default=0, metavar='N',
                        help='cuda device number (default: 0)')
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
    train_games, test_games = train_test_split(games, test_size=0.2, random_state=args.seed)
    train_games, valid_games = train_test_split(train_games, test_size=0.2, random_state=args.seed)
    train_games = pd.concat(train_games, ignore_index=True)
    valid_games = pd.concat(valid_games, ignore_index=True)
    test_games = pd.concat(test_games, ignore_index=True)

    # Dataset and dataloader
    trainset = BaseballDataset(train_games)
    validset = BaseballDataset(valid_games)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=32)
    validloader = torch.utils.data.DataLoader(validset,
        batch_size=args.batch_size, shuffle=False, num_workers=32)

    # For simulation
    valid_simul_data = valid_games[valid_games['GAME_NEW_FL']].reset_index(drop=True)
    valid_simul_set = BaseballDataset(valid_simul_data)
    valid_simul_loader = torch.utils.data.DataLoader(valid_simul_set,
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Initiate the model
    model = Model(num_bats, num_pits, num_teams, args.emb_dim, args.dropout).to(device)

    # For TensorBoard
    comment = f'_lr={args.lr}_decay={args.weight_decay}_emb_dim={args.emb_dim}_dropout={args.dropout}'
    tb = SummaryWriter(comment=comment)

    # Who wins majority
    home_wins = len(valid_simul_data[valid_simul_data['FINAL_AWAY_SCORE_CT']
        < valid_simul_data['FINAL_HOME_SCORE_CT']])
    away_wins = len(valid_simul_data[valid_simul_data['FINAL_AWAY_SCORE_CT']
        > valid_simul_data['FINAL_HOME_SCORE_CT']])
    draw = len(valid_simul_data[valid_simul_data['FINAL_AWAY_SCORE_CT']
        == valid_simul_data['FINAL_HOME_SCORE_CT']])
    total_cnt = home_wins + away_wins + draw
    print(f'home_wins={home_wins / total_cnt},'
          f'away_wins={away_wins / total_cnt},'
          f'draw={draw / total_cnt}')
    # home_wins=0.5526992287917738, away_wins=0.44473007712082263, draw=0.002570694087403599

    # tensorboard()

    if args.train_emb:
        train_embeddings(model, trainloader, validloader, device, args, tb)

    if args.load_emb:
        load_embeddings(model, tb)

    if not args.train_emb:
        train()

    tb.close()
