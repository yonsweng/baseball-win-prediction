import pandas as pd
from preprocess import preprocess
from dataset import BaseballDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from model import BaseballModel
import os.path

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

preprocessed_path = 'input/preprocessed/all2010.csv'
if os.path.exists(preprocessed_path):
    data = pd.read_csv(preprocessed_path, low_memory=False)
else:
    data = preprocess(data)
    data.to_csv(preprocessed_path, index=False)

trainset = BaseballDataset(data)
print(trainset[0:4])

writer = SummaryWriter('runs/baseball')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)

# get some random training samples
dataiter = iter(trainloader)
start_obs, end_obs, info, targets = dataiter.next()

net = BaseballModel(num_bats, num_pits, num_teams, 256)

# writer.add_graph(net, list(start_obs.values()))
writer.add_graph(net.representation, list(start_obs.values()))
writer.close()
