from torch.utils.data import Dataset
from collections.abc import Iterable


class BaseballDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            return {column: self.data[column].iloc[idx].values
                    for column in self.data.columns}
        else:
            return {column: self.data[column].at[idx]
                    for column in self.data.columns}
