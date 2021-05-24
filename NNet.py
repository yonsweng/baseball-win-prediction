import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
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
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features)
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
        x = F.relu(x, inplace=False)
        x = self.linear2(x)
        return x


class IsDone(nn.Module):
    def __init__(self, in_features, hidden_features, num_blocks):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features)
              for _ in range(num_blocks))
        )
        self.linear2 = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=False)
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
        x = self.linear1(x)
        x = F.relu(x, inplace=False)
        x = self.res_blocks(x)
        x = self.linear2(x)
        return x
