'''
model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    '''
    Model
    '''
    def __init__(self, num_bats, num_pits, num_teams, embedding_dim):
        super().__init__()

        # Layer parameters
        self.embedding_dim = embedding_dim

        # Embedding layers
        self.bat_emb = nn.Embedding(num_bats, self.embedding_dim, 0)
        self.pit_emb = nn.Embedding(num_pits, self.embedding_dim)
        self.team_emb = nn.Embedding(num_teams, self.embedding_dim)

        self.static_representation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(22 * self.embedding_dim, 4 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embedding_dim, 2 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        )

        self.compression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 8),
            nn.ReLU()
        )

        self.dynamic_representation = nn.Sequential(
            nn.Linear(self.embedding_dim // 8 + 25, 4 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embedding_dim, 2 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        )

        self.dynamics = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 2 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU()
        )

        self.next_dynamic_state = nn.Sequential(
            nn.Linear(self.embedding_dim // 2, 4 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embedding_dim, 2 * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        )

        self.reward = nn.Sequential(
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 8),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 8, 1)
        )

        self.done = nn.Sequential(
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 8),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 8, 1),
            nn.Sigmoid()
        )

        self.prediction = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 2)
        )

    def get_static_state(
        self,
        away_start_bat_ids,  # torch.long(batch_size, 9)
        home_start_bat_ids,  # torch.long(batch_size, 9)
        away_start_pit_id,  # torch.long(batch_size, 1)
        home_start_pit_id,  # torch.long(batch_size, 1)
        away_team_id,  # torch.long(batch_size, 1)
        home_team_id  # torch.long(batch_size, 1)
    ):
        away_start_bats = self.bat_emb(away_start_bat_ids)  # (batch_size, 9, emb_dim)
        home_start_bats = self.bat_emb(home_start_bat_ids)  # (batch_size, 9, emb_dim)
        away_start_pit = self.pit_emb(away_start_pit_id)  # (batch_size, 1, emb_dim)
        home_start_pit = self.pit_emb(home_start_pit_id)  # (batch_size, 1, emb_dim)
        away_team = self.team_emb(away_team_id)  # (batch_size, 1, emb_dim)
        home_team = self.team_emb(home_team_id)  # (batch_size, 1, emb_dim)
        embeddings = torch.cat([away_start_bats,
                                home_start_bats,
                                away_start_pit,
                                home_start_pit,
                                away_team,
                                home_team], dim=1)
        static_state = self.static_representation(embeddings)
        return static_state

    def get_dynamic_state(
        self,
        base1_run_id,  # torch.long(batch_size, 1)
        base2_run_id,  # torch.long(batch_size, 1)
        base3_run_id,  # torch.long(batch_size, 1)
        away_bat_lineup,  # torch.long(batch_size, 1)
        home_bat_lineup,  # torch.long(batch_size, 1)
        away_pit_lineup,  # torch.float(batch_size, 1)
        home_pit_lineup,  # torch.float(batch_size, 1)
        bat_home_id,  # torch.float(batch_size, 1)
        inn_ct,  # torch.float(batch_size, 1)
        outs_ct,  # torch.float(batch_size, 1)
        away_score_ct,  # torch.float(batch_size, 1)
        home_score_ct  # torch.float(batch_size, 1)
    ):
        '''
        Returns:
            state
        '''
        base1_run = self.bat_emb(base1_run_id)  # (batch_size, 1, emb_dim)
        base2_run = self.bat_emb(base2_run_id)  # (batch_size, 1, emb_dim)
        base3_run = self.bat_emb(base3_run_id)  # (batch_size, 1, emb_dim)
        away_bat_lineup_onehot = F.one_hot(away_bat_lineup - 1, num_classes=9).squeeze()
        home_bat_lineup_onehot = F.one_hot(home_bat_lineup - 1, num_classes=9).squeeze()

        compressed = self.compression(torch.cat([
            base1_run,
            base2_run,
            base3_run
        ], dim=1))

        representation_input = torch.cat([
            compressed,
            away_bat_lineup_onehot,
            home_bat_lineup_onehot,
            away_pit_lineup,
            home_pit_lineup,
            bat_home_id,
            inn_ct,
            outs_ct,
            away_score_ct,
            home_score_ct
        ], dim=1)

        state = self.dynamic_representation(representation_input)
        return state

    def forward(self, static_state, dynamic_state):
        '''
        Returns:
            next_dynamic_state,
            reward,
            done,
            value_away,
            value_home
        '''
        state = torch.cat([static_state, dynamic_state], dim=1)
        dynamics_output = self.dynamics(state)
        next_dynamic_state = self.next_dynamic_state(dynamics_output)
        reward = self.reward(dynamics_output)
        done = self.done(dynamics_output)

        prediction_output = self.prediction(state)
        value_away = prediction_output[:, :1]
        value_home = prediction_output[:, 1:]

        return next_dynamic_state, reward, done, value_away, value_home
