import pandas as pd
from torch.utils.data import Dataset
from collections.abc import Iterable


class BaseballDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return {column: self.data[column].iloc[idx].tolist()
                    for column in self.data.columns}
        else:
            return {column: self.data[column].at[idx]
                    for column in self.data.columns}


def get_train_data():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_train.csv',
        low_memory=False)
    return BaseballDataset(data)


def get_valid_data():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_train.csv',
        low_memory=False)
    return BaseballDataset(data)


def get_test_data():
    data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_train.csv',
        low_memory=False)
    return BaseballDataset(data)
