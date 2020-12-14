import os.path
import pandas as pd
from preprocess import preprocess
from dataset import BaseballDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from model import Model
from sklearn.model_selection import train_test_split

# Constants
MAX_EPOCHS = 100
device = torch.device('cuda:1')

# Load data
data = pd.read_csv('input/mlbplaybyplay2010s/all2010.csv', low_memory=False)

# Count the number of batters, pitchers and teams.
bats = list(data['BAT_ID']) \
     + list(data['BASE1_RUN_ID']) \
     + list(data['BASE2_RUN_ID']) \
     + list(data['BASE3_RUN_ID'])
num_bats = len(set(bats))
num_pits = len(data['PIT_ID'].unique())
num_teams = len(data['AWAY_TEAM_ID'].unique())
print(f'# of teams: {num_teams}, # of batters: {num_bats}, # of pitchers: {num_pits}')

# Load preprocessed data or make it.
preprocessed_path = 'input/preprocessed/all2010.csv'
if os.path.exists(preprocessed_path):
    data = pd.read_csv(preprocessed_path, low_memory=False)
else:
    data = preprocess(data)
    data.to_csv(preprocessed_path, index=False)

# Train-test split
games = [game for _, game in data.groupby(data['GAME_ID'])]
train_games, test_games = train_test_split(games, test_size=0.2)
train, valid = train_test_split(pd.concat(train_games, ignore_index=False), test_size=0.2)

# Dataset and dataloader
trainset = BaseballDataset(train)
validset = BaseballDataset(valid)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=32)
validloader = torch.utils.data.DataLoader(validset, batch_size=512, shuffle=True, num_workers=32)

# Initiate the model
lr = 0.01
model = Model(num_bats, num_pits, num_teams, 256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
MSELoss = torch.nn.MSELoss()
BCELoss = torch.nn.BCELoss()

# Loop over epochs
for epoch in range(MAX_EPOCHS):
    print(f'epoch {epoch}')

    training_losses = {
        'loss_state': [],
        'loss_reward': [],
        'loss_inn_end_fl': [],
        'loss_value_game': [],
        'loss_value_away': [],
        'loss_value_home': [],
        'loss': []
    }
    validation_losses = {
        'loss_state': [],
        'loss_reward': [],
        'loss_inn_end_fl': [],
        'loss_value_game': [],
        'loss_value_away': [],
        'loss_value_home': [],
        'loss': []
    }

    # Training
    for start_obs, end_obs, info, targets in trainloader:
        # Get bat_this_id and bat_next_ids
        start_bats_ids = torch.where(start_obs['bat_home_id'].repeat(1, 9) == 0,
                                     info['away_start_bat_ids'], info['home_start_bat_ids'])
        bat_lineup = start_obs['bat_lineup']
        bat_next_matrix = [(bat_lineup + i) % 9 for i in range(8)]
        bat_next_matrix = torch.cat(bat_next_matrix, dim=1)
        bat_this_id = torch.gather(start_bats_ids, 1, bat_lineup - 1)
        bat_next_ids = torch.gather(start_bats_ids, 1, bat_next_matrix)

        # Get start_pit_id and fld_team_id
        start_pit_id = torch.where(start_obs['bat_home_id'] == 0,
                                   info['home_start_pit_id'], info['away_start_pit_id'])
        fld_team_id = torch.where(start_obs['bat_home_id'] == 0,
                                  info['home_team_id'], info['away_team_id'])

        # Get state
        state = model.get_state(
            away_score_ct=start_obs['away_score_ct'].to(device),
            home_score_ct=start_obs['home_score_ct'].to(device),
            inn_ct=start_obs['inn_ct'].to(device),
            bat_home_id=start_obs['bat_home_id'].to(device),
            pit_lineup=start_obs['pit_lineup'].to(device),
            outs_ct=start_obs['outs_ct'].to(device),
            base1_run_id=start_obs['base1_run_id'].to(device),
            base2_run_id=start_obs['base2_run_id'].to(device),
            base3_run_id=start_obs['base3_run_id'].to(device)
        )

        # 모델에 어떤 값을 넣어줄지 정리하자
        next_state, reward, inn_end_fl, value_game, value_away, value_home = \
            model(start_pit_id.to(device),
                  fld_team_id.to(device),
                  state.to(device),
                  bat_this_id.to(device),
                  bat_next_ids.to(device))

        # Get next_state_target
        next_state_target = model.get_state(
            away_score_ct=end_obs['away_score_ct'].to(device),
            home_score_ct=end_obs['home_score_ct'].to(device),
            inn_ct=end_obs['inn_ct'].to(device),
            bat_home_id=end_obs['bat_home_id'].to(device),
            pit_lineup=end_obs['pit_lineup'].to(device),
            outs_ct=end_obs['outs_ct'].to(device),
            base1_run_id=end_obs['base1_run_id'].to(device),
            base2_run_id=end_obs['base2_run_id'].to(device),
            base3_run_id=end_obs['base3_run_id'].to(device)
        )

        # Loss
        loss_state = MSELoss(next_state, next_state_target)
        loss_reward = MSELoss(reward, targets['reward'].to(device))
        loss_inn_end_fl = BCELoss(inn_end_fl, targets['inn_end_fl'].to(device))
        loss_value_game = BCELoss(value_game, targets['value_game'].to(device))
        loss_value_away = MSELoss(value_away, targets['value_away'].to(device))
        loss_value_home = MSELoss(value_home, targets['value_home'].to(device))
        loss = loss_state + loss_reward + loss_inn_end_fl + \
            loss_value_game + loss_value_away + loss_value_home

        training_losses['loss_state'].append(loss_state.tolist())
        training_losses['loss_reward'].append(loss_reward.tolist())
        training_losses['loss_inn_end_fl'].append(loss_inn_end_fl.tolist())
        training_losses['loss_value_game'].append(loss_value_game.tolist())
        training_losses['loss_value_away'].append(loss_value_away.tolist())
        training_losses['loss_value_home'].append(loss_value_home.tolist())
        training_losses['loss'].append(loss.tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print losses
    for key in training_losses:
        print(f'{key}: {sum(training_losses[key]) / len(training_losses[key])}')

    # Validation
    for start_obs, end_obs, info, targets in validloader:
        # Get bat_this_id and bat_next_ids
        start_bats_ids = torch.where(start_obs['bat_home_id'].repeat(1, 9) == 0,
                                     info['away_start_bat_ids'], info['home_start_bat_ids'])
        bat_lineup = start_obs['bat_lineup']
        bat_next_matrix = [(bat_lineup + i) % 9 for i in range(8)]
        bat_next_matrix = torch.cat(bat_next_matrix, dim=1)
        bat_this_id = torch.gather(start_bats_ids, 1, bat_lineup - 1)
        bat_next_ids = torch.gather(start_bats_ids, 1, bat_next_matrix)

        # Get start_pit_id and fld_team_id
        start_pit_id = torch.where(start_obs['bat_home_id'] == 0,
                                   info['home_start_pit_id'], info['away_start_pit_id'])
        fld_team_id = torch.where(start_obs['bat_home_id'] == 0,
                                  info['home_team_id'], info['away_team_id'])

        # Get state
        state = model.get_state(
            away_score_ct=start_obs['away_score_ct'].to(device),
            home_score_ct=start_obs['home_score_ct'].to(device),
            inn_ct=start_obs['inn_ct'].to(device),
            bat_home_id=start_obs['bat_home_id'].to(device),
            pit_lineup=start_obs['pit_lineup'].to(device),
            outs_ct=start_obs['outs_ct'].to(device),
            base1_run_id=start_obs['base1_run_id'].to(device),
            base2_run_id=start_obs['base2_run_id'].to(device),
            base3_run_id=start_obs['base3_run_id'].to(device)
        )

        # 모델에 어떤 값을 넣어줄지 정리하자
        next_state, reward, inn_end_fl, value_game, value_away, value_home = \
            model(start_pit_id.to(device),
                  fld_team_id.to(device),
                  state.to(device),
                  bat_this_id.to(device),
                  bat_next_ids.to(device))

        # Get next_state_target
        next_state_target = model.get_state(
            away_score_ct=end_obs['away_score_ct'].to(device),
            home_score_ct=end_obs['home_score_ct'].to(device),
            inn_ct=end_obs['inn_ct'].to(device),
            bat_home_id=end_obs['bat_home_id'].to(device),
            pit_lineup=end_obs['pit_lineup'].to(device),
            outs_ct=end_obs['outs_ct'].to(device),
            base1_run_id=end_obs['base1_run_id'].to(device),
            base2_run_id=end_obs['base2_run_id'].to(device),
            base3_run_id=end_obs['base3_run_id'].to(device)
        )

        # Loss
        loss_state = MSELoss(next_state, next_state_target)
        loss_reward = MSELoss(reward, targets['reward'].to(device))
        loss_inn_end_fl = BCELoss(inn_end_fl, targets['inn_end_fl'].to(device))
        loss_value_game = BCELoss(value_game, targets['value_game'].to(device))
        loss_value_away = MSELoss(value_away, targets['value_away'].to(device))
        loss_value_home = MSELoss(value_home, targets['value_home'].to(device))
        loss = loss_state + loss_reward + loss_inn_end_fl + \
            loss_value_game + loss_value_away + loss_value_home

        validation_losses['loss_state'].append(loss_state.tolist())
        validation_losses['loss_reward'].append(loss_reward.tolist())
        validation_losses['loss_inn_end_fl'].append(loss_inn_end_fl.tolist())
        validation_losses['loss_value_game'].append(loss_value_game.tolist())
        validation_losses['loss_value_away'].append(loss_value_away.tolist())
        validation_losses['loss_value_home'].append(loss_value_home.tolist())
        validation_losses['loss'].append(loss.tolist())

    # Print losses
    for key in validation_losses:
        print(f'{key}: {sum(validation_losses[key]) / len(validation_losses[key])}')

# TensorBoard
# dataiter = iter(trainloader)
# start_obs, end_obs, info, targets = dataiter.next()
# writer = SummaryWriter('runs/baseball')
# writer.add_graph(net, list(start_obs.values()))
# writer.close()
