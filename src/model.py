import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_bats, num_pits, num_teams, emb_dim, dropout):
        super().__init__()

        # Model parameters
        self.emb_dim = emb_dim
        self.dropout = dropout

        # Embedding layers
        self.bat_emb = nn.Embedding(num_bats, self.emb_dim, 0)
        self.pit_emb = nn.Embedding(num_pits, self.emb_dim)
        self.team_emb = nn.Embedding(num_teams, self.emb_dim)

        # Make sub-models
        self.make_embed_model()
        self.make_dynamic_representation_model()
        self.make_dynamics_model()
        self.make_prediction_model()

    def make_embed_model(self):
        self.embed_model = nn.Sequential(
            nn.Linear(1 + 6 * self.emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.bat_dest_layer = nn.Linear(64, 5)
        self.run1_dest_layer = nn.Linear(64, 5)
        self.run2_dest_layer = nn.Linear(64, 5)
        self.run3_dest_layer = nn.Linear(64, 5)

    def make_dynamic_representation_model(self):
        return

    def make_dynamics_model(self):
        return

    def make_prediction_model(self):
        return

    def embed(
        self,
        outs_ct,
        pit_id,
        fld_team_id,
        bat_id,
        base1_run_id,
        base2_run_id,
        base3_run_id
    ):
        '''
        Returns:
            bat_dest     (BATCH_SIZE, 5),
            run1_dest_id (BATCH_SIZE, 5),
            run2_dest_id (BATCH_SIZE, 5),
            run3_dest_id (BATCH_SIZE, 5)
        '''
        pit = self.pit_emb(pit_id).squeeze()
        fld_team = self.team_emb(fld_team_id).squeeze()
        bat = self.bat_emb(bat_id).squeeze()
        base1_run = self.bat_emb(base1_run_id).squeeze()
        base2_run = self.bat_emb(base2_run_id).squeeze()
        base3_run = self.bat_emb(base3_run_id).squeeze()

        embeddings = torch.cat([
            outs_ct,
            pit,
            fld_team,
            bat,
            base1_run,
            base2_run,
            base3_run
        ], dim=1)

        out = self.embed_model(embeddings)

        bat_dest = self.bat_dest_layer(out)
        run1_dest = self.run1_dest_layer(out)
        run2_dest = self.run2_dest_layer(out)
        run3_dest = self.run3_dest_layer(out)

        return bat_dest, run1_dest, run2_dest, run3_dest

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
