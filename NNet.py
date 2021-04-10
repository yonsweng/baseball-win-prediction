import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(BasicBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_features, hidden_features)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out


class NNet(nn.Module):
    def __init__(self, n_batters, n_pitchers, n_teams,
                 embedding_dim, in_features, hidden_features, out_features):
        super(NNet, self).__init__()
        self.bat_emb = nn.Embedding(n_batters, embedding_dim)
        self.pit_emb = nn.Embedding(n_pitchers, embedding_dim)
        self.team_emb = nn.Embedding(n_teams, embedding_dim)
        self.block1 = BasicBlock(in_features, hidden_features)
        self.block2 = BasicBlock(hidden_features, hidden_features)
        self.block3 = BasicBlock(hidden_features, hidden_features)
        self.block4 = BasicBlock(hidden_features, hidden_features)
        self.linear = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = torch.cat([
            x['float'],
            torch.flatten(self.bat_emb(x['bat']), start_dim=1),
            torch.flatten(self.pit_emb(x['pit']), start_dim=1),
            torch.flatten(self.team_emb(x['team']), start_dim=1)
        ], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear(x)
        return x
