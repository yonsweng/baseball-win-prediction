import pandas as pd
from preprocess import preprocess
from dataset import BaseballDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from model import Model
import os.path
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)

# Initiate the model
model = Model(num_bats, num_pits, num_teams, 256)

# Loop over epochs
for epoch in range(MAX_EPOCHS):
    # Training
    for start_obs, end_obs, info, targets in trainloader:
        # 모델에 어떤 값을 넣어줄지 정리하자

    # Validation
    for start_obs, end_obs, info, targets in validloader:
        pass

for batch_ndx, sample in enumerate(loader):


# TensorBoard
# dataiter = iter(trainloader)
# start_obs, end_obs, info, targets = dataiter.next()
# writer = SummaryWriter('runs/baseball')
# writer.add_graph(net, list(start_obs.values()))
# writer.close()
