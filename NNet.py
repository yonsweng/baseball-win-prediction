import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features, batch_norm=False):
        super(ResBlock, self).__init__()
        self.batch_norm = batch_norm
        self.linear1 = nn.Linear(in_features, hidden_features)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_features, in_features)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        if self.batch_norm:
            out = out.permute(1, 2, 0)
            out = self.bn1(out)
            out = out.permute(2, 0, 1)
        out = self.relu(out)
        out = self.linear2(out)
        if self.batch_norm:
            out = out.permute(1, 2, 0)
            out = self.bn2(out)
            out = out.permute(2, 0, 1)
        out += residual
        out = self.relu(out)
        return out


class Represent(nn.Module):
    def __init__(self, batters, pitchers, teams, float_features, long_features,
                 out_features, embedding_dim, hidden_features, num_blocks):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.bat_emb = nn.Embedding(batters, self.embedding_dim)
        self.pit_emb = nn.Embedding(pitchers, self.embedding_dim)
        self.team_emb = nn.Embedding(teams, self.embedding_dim)

        in_features = (float_features + long_features) * self.embedding_dim
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features, batch_norm=True)
              for _ in range(num_blocks))
        )
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: dict[str, torch.Tensor]):
        x = torch.cat([
            x['float'].repeat(1, 1, self.embedding_dim),
            torch.flatten(self.bat_emb(x['bat']), start_dim=2),
            torch.flatten(self.pit_emb(x['pit']), start_dim=2),
            torch.flatten(self.team_emb(x['team']), start_dim=2)
        ], dim=2)
        x = self.linear1(x)
        x = x.permute(1, 2, 0)
        x = self.bn1(x)
        x = x.permute(2, 0, 1)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        x = self.linear2(x)
        return x


class Dynamics(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_blocks):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_size, hidden_size, batch_norm=True)
              for _ in range(num_blocks))
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, h_0):
        '''
        x: (seq_len, batch_size, represent_dim)
        '''
        x, h_n = self.rnn(x, h_0)
        if not torch.is_tensor(x):
            x, lengths = pad_packed_sequence(x)
        else:
            lengths = None
        x = self.res_blocks(x)
        x = self.linear(x)
        return x, h_n, lengths


class IsDone(nn.Module):
    def __init__(self, in_features, hidden_features, num_blocks):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features, batch_norm=True)
              for _ in range(num_blocks))
        )
        self.linear2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = x.permute(1, 2, 0)
        x = self.bn1(x)
        x = x.permute(2, 0, 1)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class Predict(nn.Module):
    def __init__(self, in_features, hidden_features, num_blocks):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features)
              for _ in range(num_blocks))
        )
        self.linear2 = nn.Linear(hidden_features, 2)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        x = self.linear2(x)
        x = x.squeeze(0)
        return x
