import torch
from torch.utils.data import Dataset


class BaseballDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :]

        features = {
            'away_score_ct': torch.Tensor([data['AWAY_SCORE_CT']]),
            'home_score_ct': torch.Tensor([data['HOME_SCORE_CT']]),
            'inn_ct': torch.Tensor([data['INN_CT']]),
            'bat_home_id': torch.Tensor([data['BAT_HOME_ID']]),
            'outs_ct': torch.Tensor([data['OUTS_CT']]),
            'bat_id': torch.tensor([data['BAT_ID']], dtype=torch.long),
            'pit_id': torch.tensor([data['PIT_ID']], dtype=torch.long),
            'base1_run_id': torch.tensor([data['BASE1_RUN_ID']], dtype=torch.long),
            'base2_run_id': torch.tensor([data['BASE2_RUN_ID']], dtype=torch.long),
            'base3_run_id': torch.tensor([data['BASE3_RUN_ID']], dtype=torch.long),
            'away_bat_lineup': torch.tensor([data['AWAY_BAT_LINEUP_ID']], dtype=torch.long),
            'home_bat_lineup': torch.tensor([data['HOME_BAT_LINEUP_ID']], dtype=torch.long),
            'away_start_bat_ids': torch.from_numpy(
                data[[f'AWAY_START_BAT{i}_ID' for i in range(1, 10)]].values.astype(int)),
            'home_start_bat_ids': torch.from_numpy(
                data[[f'HOME_START_BAT{i}_ID' for i in range(1, 10)]].values.astype(int)),
            'away_start_pit_id': torch.tensor([data['AWAY_START_PIT_ID']], dtype=torch.long),
            'home_start_pit_id': torch.tensor([data['HOME_START_PIT_ID']], dtype=torch.long),
            'away_team_id': torch.tensor([data['AWAY_TEAM_ID']], dtype=torch.long),
            'home_team_id': torch.tensor([data['HOME_TEAM_ID']], dtype=torch.long)
        }
        targets = {
            'event_runs_ct': torch.Tensor([data['EVENT_RUNS_CT']]),
            'event_outs_ct': torch.Tensor([data['EVENT_OUTS_CT']]),
            'bat_dest': torch.tensor([data['BAT_DEST_ID']], dtype=torch.long),
            'run1_dest': torch.tensor([data['RUN1_DEST_ID']], dtype=torch.long),
            'run2_dest': torch.tensor([data['RUN2_DEST_ID']], dtype=torch.long),
            'run3_dest': torch.tensor([data['RUN3_DEST_ID']], dtype=torch.long),
            'value_away': torch.Tensor([data['VALUE_AWAY']]),
            'value_home': torch.Tensor([data['VALUE_HOME']])
        }

        return features, targets
