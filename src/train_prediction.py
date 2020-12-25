import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import init, load_data, count_numbers, get_dataloaders, get_latest_file_path, get_next_bats
from models import Dynamics, Prediction


def train():
    # Load pretrained weights
    path = get_latest_file_path('../models/dynamics')
    dynamics = Dynamics(num_bats, num_pits, num_teams, args.emb_dim, args.dropout)
    dynamics.load_state_dict(torch.load(path))
    model.bat_emb.load_state_dict(dynamics.bat_emb.state_dict())
    model.pit_emb.load_state_dict(dynamics.pit_emb.state_dict())
    model.team_emb.load_state_dict(dynamics.team_emb.state_dict())
    model.dense.load_state_dict(dynamics.dense.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    MSELoss = torch.nn.MSELoss()
    CELoss = torch.nn.CrossEntropyLoss()

    lr_lambda = lambda x: x / args.warmup if x <= args.warmup else (x / args.warmup) ** -0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    clip = lambda x: x.where(x <= 4, torch.tensor([4], dtype=torch.long))

    best_loss = 99.9
    early_stopping = 0

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        # Training
        model.train()
        sum_loss = 0.
        negative = 0
        positive = 0
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
            value_away, value_home = model(
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
            negative += (targets['value_away'] > targets['value_home']).sum().item()
            positive += (targets['value_away'] < targets['value_home']).sum().item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        tb.add_scalar('train loss', sum_loss / len(trainloader.dataset), epoch)
        tb.add_scalar('train TNR', true_negative / negative, epoch)
        tb.add_scalar('train TPR', true_positive / positive, epoch)
        tb.add_scalar('train acc.', (true_negative + true_positive) / (positive + negative), epoch)
        tb.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # Validation
        model.eval()
        sum_loss = 0.
        negative = 0
        positive = 0
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
            value_away, value_home = model(
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
            negative += (targets['value_away'] > targets['value_home']).sum().item()
            positive += (targets['value_away'] < targets['value_home']).sum().item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
        tb.add_scalar('valid loss', sum_loss / len(validloader.dataset), epoch)
        tb.add_scalar('valid TNR', true_negative / negative, epoch)
        tb.add_scalar('valid TPR', true_positive / positive, epoch)
        tb.add_scalar('valid acc.', (true_negative + true_positive) / (positive + negative), epoch)

        negative = 0
        positive = 0
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
            value_away, value_home = model(
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
            negative += (targets['value_away'] > targets['value_home']).sum().item()
            positive += (targets['value_away'] < targets['value_home']).sum().item()
            true_negative += torch.logical_and(value_away.cpu() > value_home.cpu(),
                targets['value_away'] > targets['value_home']).sum().item()
            true_positive += torch.logical_and(value_away.cpu() < value_home.cpu(),
                targets['value_away'] < targets['value_home']).sum().item()
        tb.add_scalar('vnew TNR', true_negative / negative, epoch)
        tb.add_scalar('vnew TPR', true_positive / positive, epoch)
        tb.add_scalar('vnew acc.', (true_negative + true_positive) / (positive + negative), epoch)

        # Save the best model.
        if sum_loss / len(validloader.dataset) < best_loss:
            best_loss = sum_loss / len(validloader.dataset)
            torch.save(model.state_dict(), f'../models/{PREFIX}/{tag}.pt')
            print('model saved.')
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt > args.patience:
                break


if __name__ == "__main__":
    PREFIX = 'prediction'
    parser = argparse.ArgumentParser()  # 자주 바뀌는 순.
    parser.add_argument('--dropout', type=float, default=0.2, metavar='F')
    parser.add_argument('--l2', type=float, default=1e-3, metavar='F')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='F')
    parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
    parser.add_argument('--warmup', type=int, default=1000, metavar='N')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N')
    parser.add_argument('--epochs', type=int, default=50, metavar='N')
    parser.add_argument('--patience', type=int, default=3, metavar='N')
    parser.add_argument('--seed', type=int, default=777, metavar='N')
    parser.add_argument('--workers', type=int, default=16, metavar='N')
    parser.add_argument('--cuda', type=int, default=1, metavar='N')
    args = parser.parse_args()

    tag, device = init(args)
    data = load_data()
    num_bats, num_pits, num_teams = count_numbers(data)
    trainloader, validloader, vnewloader = get_dataloaders(data, args)

    model = Prediction(num_bats, num_pits, num_teams, args.emb_dim, args.dropout).to(device)
    tb = SummaryWriter(f'../runs/{PREFIX}_{tag}')
    train()
    tb.close()
