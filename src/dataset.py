import torch
from torch.utils.data import Dataset
import numpy as np


class BaseballDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :]

        start_obs = {
            'inn_ct': torch.Tensor([data['INN_CT']]),
            'bat_home_id': torch.Tensor([data['BAT_HOME_ID']]),
            'outs_ct': torch.Tensor([data['OUTS_CT']]),
            'away_score_ct': torch.Tensor([data['AWAY_SCORE_CT']]),
            'home_score_ct': torch.Tensor([data['HOME_SCORE_CT']]),
            'base1_run_id': torch.tensor([data['BASE1_RUN_ID']], dtype=torch.long),
            'base2_run_id': torch.tensor([data['BASE2_RUN_ID']], dtype=torch.long),
            'base3_run_id': torch.tensor([data['BASE3_RUN_ID']], dtype=torch.long),
            'bat_lineup': torch.tensor([data['BAT_LINEUP_ID']], dtype=torch.long),
            'pit_lineup': torch.Tensor([data['PIT_LINEUP_ID']])
        }
        end_obs = {
            'inn_ct': torch.Tensor([data['END_INN_CT']]),
            'bat_home_id': torch.Tensor([data['END_BAT_HOME_ID']]),
            'outs_ct': torch.Tensor([data['END_OUTS_CT']]),
            'away_score_ct': torch.Tensor([data['END_AWAY_SCORE_CT']]),
            'home_score_ct': torch.Tensor([data['END_HOME_SCORE_CT']]),
            'base1_run_id': torch.tensor([data['END_BASE1_RUN_ID']], dtype=torch.long),
            'base2_run_id': torch.tensor([data['END_BASE2_RUN_ID']], dtype=torch.long),
            'base3_run_id': torch.tensor([data['END_BASE3_RUN_ID']], dtype=torch.long),
            'bat_lineup': torch.tensor([data['END_BAT_LINEUP_ID']], dtype=torch.long),
            'pit_lineup': torch.Tensor([data['END_PIT_LINEUP_ID']])
        }
        info = {
            'away_start_pit_id': torch.tensor([data['AWAY_START_PIT_ID']], dtype=torch.long),
            'home_start_pit_id': torch.tensor([data['HOME_START_PIT_ID']], dtype=torch.long),
            'away_team_id': torch.tensor([data['AWAY_TEAM_ID']], dtype=torch.long),
            'home_team_id': torch.tensor([data['HOME_TEAM_ID']], dtype=torch.long),
            'away_start_bat_ids': torch.from_numpy(
                data[[f'AWAY_START_BAT{i}_ID' for i in range(1, 10)]].values.astype(int)),
            'home_start_bat_ids': torch.from_numpy(
                data[[f'HOME_START_BAT{i}_ID' for i in range(1, 10)]].values.astype(int))
        }
        targets = {
            'inn_end_fl': torch.Tensor([data['INN_END_FL']]),
            'reward': torch.Tensor([data['END_AWAY_SCORE_CT'] - data['AWAY_SCORE_CT'] if data['BAT_HOME_ID'] == 0
                               else data['END_HOME_SCORE_CT'] - data['HOME_SCORE_CT']]),
            'value_game': torch.Tensor([0.5 if data['AWAY_SCORE_CT'] == data['HOME_SCORE_CT']
                                    else float(data['AWAY_SCORE_CT'] < data['HOME_SCORE_CT'])]),
            'value_away': torch.Tensor([data['FINAL_AWAY_SCORE_CT'] - data['AWAY_SCORE_CT']]),
            'value_home': torch.Tensor([data['FINAL_HOME_SCORE_CT'] - data['HOME_SCORE_CT']])
        }

        return start_obs, end_obs, info, targets
