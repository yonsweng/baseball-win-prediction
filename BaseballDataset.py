import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PlateDataset(Dataset):
    def __init__(self, data, event_to_index):
        super().__init__()
        self.data = data
        self.event_to_index = event_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        if row['BAT_HOME_ID'] == 0:  # away team batting
            bat_id = row['AWAY_BAT1_ID']
            pit_id = row['HOME_PIT_ID']
            team_id = row['HOME_TEAM_ID']
        else:  # home team batting
            bat_id = row['HOME_BAT1_ID']
            pit_id = row['AWAY_PIT_ID']
            team_id = row['AWAY_TEAM_ID']

        return torch.tensor([bat_id, pit_id, team_id], dtype=torch.long), \
            torch.tensor(
                self.event_to_index[row['EVENT_CD']],
                dtype=torch.long
            )


class BaseballDataset(Dataset):
    def __init__(self, data):
        # Split data by games
        self.games = []
        game_new_indice = list(data[data['GAME_NEW_FL'] == 'T'].index) \
            + [len(data.index)]
        for i in range(len(game_new_indice) - 1):
            game = data.iloc[game_new_indice[i]:game_new_indice[i+1]]
            self.games.append(game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]

        features = {
            'float': torch.tensor([
                game['INN_CT'].values / 9,  # 1/9, 2/9, ..., 1, ...
                (game['BAT_HOME_ID'].values - 0.5) * 2,  # -1, 1
                (game['OUTS_CT'].values + 1) / 3,  # 1/3, 2/3, 1
                game['AWAY_SCORE_CT'].values / 10,
                game['HOME_SCORE_CT'].values / 10
            ], dtype=torch.float).transpose(0, 1),
            'bat': torch.tensor([
                # game['BASE1_RUN_ID'].values,
                # game['BASE2_RUN_ID'].values,
                # game['BASE3_RUN_ID'].values,
                *[game[f'AWAY_BAT{i}_ID'].values for i in range(1, 10)],
                *[game[f'HOME_BAT{i}_ID'].values for i in range(1, 10)]
            ], dtype=torch.long).transpose(0, 1),
            'pit': torch.tensor([
                game['AWAY_PIT_ID'].values,
                game['HOME_PIT_ID'].values
            ], dtype=torch.long).transpose(0, 1),
            'team': torch.tensor([
                game['AWAY_TEAM_ID'].values,
                game['HOME_TEAM_ID'].values
            ], dtype=torch.long).transpose(0, 1)
        }

        targets = torch.tensor([
            game['AWAY_END_SCORE_CT'].values[-1],
            game['HOME_END_SCORE_CT'].values[-1],
            game['AWAY_BAT_INN'].values[-1],
            game['HOME_BAT_INN'].values[-1]
        ], dtype=torch.float)  # (2, )

        return features, targets


def collate_fn(data):
    '''
    data: a list of tuples
    '''
    features, score_targets = zip(*data)

    lengths = [len(seq['float']) for seq in features]
    max_seq_len = max(lengths)
    batch_size = len(features)

    # Make feature_batch
    feature_batch = {
        key: pad_sequence([feature[key] for feature in features])
        for key in features[0].keys()
    }

    # Make done_batch
    done_batch = torch.zeros((max_seq_len, batch_size, 1), dtype=torch.float)
    for i, seq in enumerate(features):
        done_batch[len(seq['float'])-1, i] = 1.

    # Make score_batch
    targets = torch.stack(score_targets)
    score_batch = targets[:, :2]  # (BATCH_SIZE, 2)
    inns = targets[:, 2:]

    return feature_batch, done_batch, score_batch, lengths, inns
