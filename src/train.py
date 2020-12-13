import os.path
import pandas as pd
from preprocess import preprocess
from dataset import BaseballDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from model import Model
from sklearn.model_selection import train_test_split

# Constants
MAX_EPOCHS = 10

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True, num_workers=8)

# Initiate the model
lr = 0.001
model = Model(num_bats, num_pits, num_teams, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
MSELoss = torch.nn.MSELoss()
BCELoss = torch.nn.BCELoss()

# Loop over epochs
for epoch in range(MAX_EPOCHS):
    # Training
    for start_obs, end_obs, info, targets in trainloader:
        # Get bat_this_id and bat_next_ids
        start_bats_ids = torch.where(start_obs['bat_home_id'].unsqueeze(1).repeat(1, 9) == 0,
                                     info['away_start_bat_ids'], info['home_start_bat_ids'])
        bat_lineup = start_obs['bat_lineup']
        bat_next_matrix = [(bat_lineup + i) % 9 for i in range(8)]
        bat_next_matrix = torch.cat(bat_next_matrix, dim=1)
        bat_this_id = torch.gather(start_bats_ids, 1, bat_lineup.unsqueeze(1) - 1)
        bat_next_ids = torch.gather(start_bats_ids, 1, bat_next_matrix)

        # Get start_pit_id and fld_team_id
        start_pit_id = torch.where(start_obs['bat_home_id'] == 0,
                                   info['away_start_pit_id'], info['home_start_pit_id'])
        fld_team_id = torch.where(start_obs['bat_home_id'] == 0,
                                  info['away_fld_team_id'], info['home_fld_team_id'])

        # Get state
        state = model.get_state(
            away_score_ct=start_obs['away_score_ct'],
            home_score_ct=start_obs['home_score_ct'],
            inn_ct=start_obs['inn_ct'],
            bat_home_id=start_obs['bat_home_id'],
            pit_lineup=start_obs['pit_lineup'],
            outs_ct=start_obs['outs_ct'],
            base1_run_id=start_obs['base1_run_id'],
            base2_run_id=start_obs['base2_run_id'],
            base3_run_id=start_obs['base3_run_id']
        )

        # 모델에 어떤 값을 넣어줄지 정리하자
        next_state, reward, inn_end_fl, value_game, value_away, value_home = \
            model(start_pit_id, fld_team_id, state, bat_this_id, bat_next_ids)

        # Get next_state_target
        next_state_target = model.get_state(
            away_score_ct=end_obs['away_score_ct'],
            home_score_ct=end_obs['home_score_ct'],
            inn_ct=end_obs['inn_ct'],
            bat_home_id=end_obs['bat_home_id'],
            pit_lineup=end_obs['pit_lineup'],
            outs_ct=end_obs['outs_ct'],
            base1_run_id=end_obs['base1_run_id'],
            base2_run_id=end_obs['base2_run_id'],
            base3_run_id=end_obs['base3_run_id']
        )

        # Loss
        loss_state = MSELoss(next_state, next_state_target)
        loss_reward = MSELoss(reward, targets['reward'])
        loss_inn_end_fl = BCELoss(inn_end_fl, end_obs['inn_end_fl'])
        loss_value_game = BCELoss(value_game, targets['value_game'])
        loss_value_away = MSELoss(value_game, targets['value_away'])
        loss_value_home = MSELoss(value_game, targets['value_home'])
        loss = loss_state + loss_reward + loss_inn_end_fl + \
            loss_value_game + loss_value_away + loss_value_home

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    # for start_obs, end_obs, info, targets in validloader:
    #     pass


# TensorBoard
# dataiter = iter(trainloader)
# start_obs, end_obs, info, targets = dataiter.next()
# writer = SummaryWriter('runs/baseball')
# writer.add_graph(net, list(start_obs.values()))
# writer.close()
