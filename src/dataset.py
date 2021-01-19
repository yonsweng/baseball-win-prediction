import torch
from torch.utils.data import Dataset


class BaseballDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :]

        policy_state = {
            'outs_ct': torch.tensor([data['OUTS_CT']], dtype=torch.float),
            'bat_id': torch.tensor([data['BAT_ID']], dtype=torch.long),
            'pit_id': torch.tensor([data['PIT_ID']], dtype=torch.long),
            'fld_team_id': torch.tensor([data['AWAY_TEAM_ID'] if data['BAT_HOME_ID'] == 1 else data['HOME_TEAM_ID']], dtype=torch.long),
            'base1_run_id': torch.tensor([data['BASE1_RUN_ID']], dtype=torch.long),
            'base2_run_id': torch.tensor([data['BASE2_RUN_ID']], dtype=torch.long),
            'base3_run_id': torch.tensor([data['BASE3_RUN_ID']], dtype=torch.long)
        }
        value_state = {
            'away_score_ct': torch.tensor([data['AWAY_SCORE_CT']], dtype=torch.float),
            'home_score_ct': torch.tensor([data['HOME_SCORE_CT']], dtype=torch.float),
            'inn_ct': torch.tensor([data['INN_CT']], dtype=torch.float),
            'bat_home_id': torch.tensor([data['BAT_HOME_ID']], dtype=torch.float),
            'away_bat_lineup': torch.tensor([data['AWAY_BAT_LINEUP_ID']], dtype=torch.float),
            'home_bat_lineup': torch.tensor([data['HOME_BAT_LINEUP_ID']], dtype=torch.float),
            'away_start_bat_ids': torch.tensor(data[[f'AWAY_START_BAT{i}_ID' for i in range(1, 10)]].values.astype(int), dtype=torch.long),
            'home_start_bat_ids': torch.tensor(data[[f'HOME_START_BAT{i}_ID' for i in range(1, 10)]].values.astype(int), dtype=torch.long),
            'away_start_pit_id': torch.tensor([data['AWAY_START_PIT_ID']], dtype=torch.long),
            'home_start_pit_id': torch.tensor([data['HOME_START_PIT_ID']], dtype=torch.long),
            'away_team_id': torch.tensor([data['AWAY_TEAM_ID']], dtype=torch.long),
            'home_team_id': torch.tensor([data['HOME_TEAM_ID']], dtype=torch.long)
        }

        policy_targets = {
            'bat_dest': torch.tensor([data['BAT_DEST_ID']], dtype=torch.long),
            'run1_dest': torch.tensor([data['RUN1_DEST_ID']], dtype=torch.long),
            'run2_dest': torch.tensor([data['RUN2_DEST_ID']], dtype=torch.long),
            'run3_dest': torch.tensor([data['RUN3_DEST_ID']], dtype=torch.long)
        }
        value_target = {
            'value': torch.tensor([data['VALUE_AWAY'] + data['AWAY_SCORE_CT'] < data['VALUE_HOME'] + data['HOME_SCORE_CT']], dtype=torch.float)
        }

        return policy_state, value_state, policy_targets, value_target
