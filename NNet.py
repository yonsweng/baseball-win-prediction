import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
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

        self.bat_emb = nn.Embedding(batters, embedding_dim)
        self.pit_emb = nn.Embedding(pitchers, embedding_dim)
        self.team_emb = nn.Embedding(teams, embedding_dim)

        in_features = float_features + long_features * embedding_dim
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.res_blocks = nn.Sequential(
            *(ResBlock(hidden_features, hidden_features)
              for _ in range(num_blocks))
        )
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: dict[str, torch.Tensor]):
        x = torch.cat([
            x['float'],
            torch.flatten(self.bat_emb(x['bat']), start_dim=1),
            torch.flatten(self.pit_emb(x['pit']), start_dim=1),
            torch.flatten(self.team_emb(x['team']), start_dim=1)
        ], dim=1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
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
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        x = self.linear2(x)
        x = F.sigmoid(x)
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
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        x = self.linear2(x)
        return x


class NNet(nn.Module):
    def __init__(self, n_batters, n_pitchers, n_teams,
                 float_features, long_features, policy_dim,
                 emb_dim=64, hidden_features=512, res_features=512,
                 n_shared_layers=2, n_policy_layers=2, n_value_layers=2):
        super(NNet, self).__init__()
        self.bat_emb = nn.Embedding(n_batters, emb_dim)
        self.pit_emb = nn.Embedding(n_pitchers, emb_dim)
        self.team_emb = nn.Embedding(n_teams, emb_dim)
        self.shared_layers = nn.Sequential(
            nn.Linear(float_features+long_features*emb_dim, hidden_features),
            nn.ReLU(inplace=True),
            *[ResBlock(hidden_features, res_features)
              for _ in range(n_shared_layers)]
        )
        self.policy_layers = nn.Sequential(
            *[ResBlock(hidden_features, res_features)
              for _ in range(n_policy_layers)],
            nn.Linear(hidden_features, policy_dim)
        )
        self.value_layers = nn.Sequential(
            *[ResBlock(hidden_features, res_features)
              for _ in range(n_value_layers)],
            nn.Linear(hidden_features, 2)
        )

    def forward(self, x: dict[str, torch.Tensor]):
        x = torch.cat([
            x['float'],
            torch.flatten(self.bat_emb(x['bat']), start_dim=1),
            torch.flatten(self.pit_emb(x['pit']), start_dim=1),
            torch.flatten(self.team_emb(x['team']), start_dim=1)
        ], dim=1)
        x = self.shared_layers(x)
        return self.policy_layers(x), self.value_layers(x)

    def predict(self, x: dict[str, torch.Tensor]):
        '''
        Non-batch prediction to give numpy arrays.
        Return:
            policy: np.array(n_actions)
            value: (float, float)
        '''
        policy, value = self.forward(x)
        policy = F.softmax(policy, dim=1)
        value = value.detach().cpu().numpy().squeeze()
        return (policy.detach().cpu().numpy().squeeze(),
                (value[0], value[1]))
