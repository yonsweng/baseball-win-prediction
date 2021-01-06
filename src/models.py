import torch
import torch.nn as nn


class BatEmb(nn.Module):
    def __init__(self, num, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num, emb_dim)

    def forward(self, x):
        return self.embedding(x)


class PitEmb(nn.Module):
    def __init__(self, num, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num, emb_dim)

    def forward(self, x):
        return self.embedding(x)


class TeamEmb(nn.Module):
    def __init__(self, num, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num, emb_dim)

    def forward(self, x):
        return self.embedding(x)


class Dense(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.dense(x)


class Dynamics(nn.Module):
    def __init__(self, num_bats, num_pits, num_teams, emb_dim, dropout):
        super().__init__()

        # Embedding layers
        self.bat_emb = BatEmb(num_bats, emb_dim)
        self.pit_emb = PitEmb(num_pits, emb_dim)
        self.team_emb = TeamEmb(num_teams, emb_dim)

        # Input layers
        self.plate_in = nn.Sequential(
            nn.Linear(5 + 6 * emb_dim, 8192),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Shared layers
        self.dense = Dense(dropout)

        # Output layers
        self.bat_dest = nn.Linear(512, 5)
        self.run1_dest = nn.Linear(512, 5)
        self.run2_dest = nn.Linear(512, 5)
        self.run3_dest = nn.Linear(512, 5)

        # Rewards
        self.rewards = []
        self.saved_log_probs = []

    def forward(
        self,
        away_score_ct,  # (BATCH_SIZE, 1)
        home_score_ct,
        inn_ct,
        bat_home_id,
        outs_ct,
        bat_id,
        start_pit_id,
        fld_team_id,
        base1_run_id,
        base2_run_id,
        base3_run_id
    ):
        '''
        Returns:
            bat_dest      (BATCH_SIZE, 5),
            run1_dest     (BATCH_SIZE, 5),
            run2_dest     (BATCH_SIZE, 5),
            run3_dest     (BATCH_SIZE, 5)
        '''
        bat = self.bat_emb(bat_id).reshape(bat_id.shape[0], -1)
        start_pit = self.pit_emb(start_pit_id).reshape(start_pit_id.shape[0], -1)
        fld_team = self.team_emb(fld_team_id).reshape(fld_team_id.shape[0], -1)
        base1_run = self.bat_emb(base1_run_id).reshape(base1_run_id.shape[0], -1)
        base2_run = self.bat_emb(base2_run_id).reshape(base2_run_id.shape[0], -1)
        base3_run = self.bat_emb(base3_run_id).reshape(base3_run_id.shape[0], -1)

        values = torch.cat([
            away_score_ct,  # (BATCH_SIZE, 1)
            home_score_ct,
            inn_ct,
            bat_home_id,
            outs_ct,
            bat,            # (BATCH_SIZE, emb_dim)
            start_pit,
            fld_team,
            base1_run,
            base2_run,
            base3_run
        ], dim=1)

        values = self.plate_in(values)

        values = self.dense(values)

        bat_dest = self.bat_dest(values)
        run1_dest = self.run1_dest(values)
        run2_dest = self.run2_dest(values)
        run3_dest = self.run3_dest(values)

        return bat_dest, run1_dest, run2_dest, run3_dest


class Prediction(nn.Module):
    def __init__(self, num_bats, num_pits, num_teams, emb_dim, dropout):
        super().__init__()

        # Embedding layers
        self.bat_emb = BatEmb(num_bats, emb_dim)
        self.pit_emb = PitEmb(num_pits, emb_dim)
        self.team_emb = TeamEmb(num_teams, emb_dim)

        # Input layers
        self.game_in = nn.Sequential(
            nn.Linear(5 + 28 * emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Shared layers
        self.dense = Dense(dropout)

        # Output layers
        self.value_away = nn.Linear(128, 1)
        self.value_home = nn.Linear(128, 1)

    def forward(
        self,
        away_score_ct,        # (BATCH_SIZE, 1)
        home_score_ct,
        inn_ct,
        bat_home_id,
        outs_ct,
        bat_id,
        start_pit_id,
        fld_team_id,
        base1_run_id,
        base2_run_id,
        base3_run_id,
        away_start_bats_ids,  # (BATCH_SIZE, 9)
        home_start_bats_ids,
        away_start_pit_id,    # (BATCH_SIZE, 1)
        home_start_pit_id,
        away_team_id,
        home_team_id
    ):
        '''
        Returns:
            value_away (BATCH_SIZE, 1),
            value_home (BATCH_SIZE, 1)
        '''
        bat = self.bat_emb(bat_id).squeeze()
        start_pit = self.pit_emb(start_pit_id).squeeze()
        fld_team = self.team_emb(fld_team_id).squeeze()
        base1_run = self.bat_emb(base1_run_id).squeeze()
        base2_run = self.bat_emb(base2_run_id).squeeze()
        base3_run = self.bat_emb(base3_run_id).squeeze()
        away_start_bats = self.bat_emb(away_start_bats_ids).reshape(
            away_start_bats_ids.shape[0], -1)
        home_start_bats = self.bat_emb(home_start_bats_ids).reshape(
            home_start_bats_ids.shape[0], -1)
        away_start_pit = self.pit_emb(away_start_pit_id).squeeze()
        home_start_pit = self.pit_emb(home_start_pit_id).squeeze()
        away_team = self.team_emb(away_team_id).squeeze()
        home_team = self.team_emb(home_team_id).squeeze()

        values = torch.cat([
            away_score_ct,    # (BATCH_SIZE, 1)
            home_score_ct,
            inn_ct,
            bat_home_id,
            outs_ct,
            bat,              # (BATCH_SIZE, emb_dim)
            start_pit,
            fld_team,
            base1_run,
            base2_run,
            base3_run,
            away_start_bats,  # (BATCH_SIZE, 9 * emb_dim)
            home_start_bats,
            away_start_pit,   # (BATCH_SIZE, emb_dim)
            home_start_pit,
            away_team,
            home_team
        ], dim=1)

        values = self.game_in(values)

        values = self.dense(values)

        value_away = self.value_away(values)
        value_home = self.value_home(values)

        return value_away, value_home
