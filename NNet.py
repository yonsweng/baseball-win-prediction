import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EventPredict(nn.Module):
    def __init__(self, batters, pitchers, teams, embedding_dim, out_features,
                 hidden_dim, num_linears):
        super().__init__()

        self.bat_emb = nn.Embedding(batters, embedding_dim)
        self.pit_emb = nn.Embedding(pitchers, embedding_dim)
        self.team_emb = nn.Embedding(teams, embedding_dim)

        self.linear_in = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.linears = nn.Sequential(
            *([nn.Linear(hidden_dim, hidden_dim),
               nn.BatchNorm1d(hidden_dim),
               nn.ReLU(inplace=True)] * num_linears)
        )
        self.linear_out = nn.Linear(hidden_dim, out_features)

    def forward(self, x):  # x = (batch_size, 3)
        bat = self.bat_emb(x[:, 0])
        pit = self.pit_emb(x[:, 1])
        team = self.team_emb(x[:, 2])
        x = torch.cat((bat, pit, team), dim=-1)
        x = self.linear_in(x)
        x = self.linears(x)
        x = self.linear_out(x)
        return x


class Predict(nn.Module):
    def __init__(self, embedding_dim, batters, pitchers, teams,
                 float_features, long_features, hidden_features,
                 num_blocks, num_linears):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.bat_emb = nn.Embedding(batters, self.embedding_dim)
        self.pit_emb = nn.Embedding(pitchers, self.embedding_dim)
        self.team_emb = nn.Embedding(teams, self.embedding_dim)

        # in_features = (float_features + long_features) * self.embedding_dim
        in_features = long_features * self.embedding_dim
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)

        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features)
              for _ in range(num_blocks))
        )

        self.linear_blocks = nn.Sequential(
            *((nn.Linear(hidden_features, hidden_features),
               nn.BatchNorm1d(hidden_features),
               nn.ReLU(inplace=True)) * num_linears)
        )

        self.linear_out = nn.Linear(hidden_features, 2)

    def forward(self, x):
        x = torch.cat([
            # x['float'].repeat(1, self.embedding_dim),
            torch.flatten(self.bat_emb(x['bat']), start_dim=1),
            torch.flatten(self.pit_emb(x['pit']), start_dim=1),
            torch.flatten(self.team_emb(x['team']), start_dim=1)
        ], dim=1)  # (batch_size, #features)
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        x = self.linear_blocks(x)
        x = self.linear_out(x)
        return x
