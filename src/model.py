import torch
import torch.nn as nn


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

        # Input layers
        self.small_in = nn.Sequential(
            nn.Linear(5 + 6 * self.emb_dim, 512),
            nn.ReLU()
        )
        self.big_in = nn.Sequential(
            nn.Linear(5 + 28 * self.emb_dim, 512),
            nn.ReLU()
        )

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Output layers
        self.bat_dest = nn.Linear(128, 5)
        self.run1_dest = nn.Linear(128, 5)
        self.run2_dest = nn.Linear(128, 5)
        self.run3_dest = nn.Linear(128, 5)
        self.event_outs_ct = nn.Linear(128, 1)
        self.event_runs_ct = nn.Linear(128, 1)
        self.value_away = nn.Linear(128, 1)
        self.value_home = nn.Linear(128, 1)

    def dynamics(
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
            event_runs_ct (BATCH_SIZE, 1),
            event_outs_ct (BATCH_SIZE, 1),
            bat_dest      (BATCH_SIZE, 5),
            run1_dest     (BATCH_SIZE, 5),
            run2_dest     (BATCH_SIZE, 5),
            run3_dest     (BATCH_SIZE, 5)
        '''
        bat = self.bat_emb(bat_id).squeeze()
        start_pit = self.pit_emb(start_pit_id).squeeze()
        fld_team = self.team_emb(fld_team_id).squeeze()
        base1_run = self.bat_emb(base1_run_id).squeeze()
        base2_run = self.bat_emb(base2_run_id).squeeze()
        base3_run = self.bat_emb(base3_run_id).squeeze()

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

        values = self.small_in(values)

        values = self.shared(values)

        event_runs_ct = self.event_runs_ct(values)
        event_outs_ct = self.event_outs_ct(values)
        bat_dest = self.bat_dest(values)
        run1_dest = self.run1_dest(values)
        run2_dest = self.run2_dest(values)
        run3_dest = self.run3_dest(values)

        return event_runs_ct, event_outs_ct, \
            bat_dest, run1_dest, run2_dest, run3_dest

    def prediction(
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
        away_start_bats = self.bat_emb(away_start_bats_ids).reshape(-1, 9 * self.emb_dim)
        home_start_bats = self.bat_emb(home_start_bats_ids).reshape(-1, 9 * self.emb_dim)
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

        values = self.big_in(values)

        values = self.shared(values)

        value_away = self.value_away(values)
        value_home = self.value_home(values)

        return value_away, value_home

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
            event_runs_ct (BATCH_SIZE, 1),
            event_outs_ct (BATCH_SIZE, 1),
            bat_dest      (BATCH_SIZE, 5),
            run1_dest     (BATCH_SIZE, 5),
            run2_dest     (BATCH_SIZE, 5),
            run3_dest     (BATCH_SIZE, 5)
            value_away (BATCH_SIZE, 1),
            value_home (BATCH_SIZE, 1)
        '''
        bat = self.bat_emb(bat_id).squeeze()
        start_pit = self.pit_emb(start_pit_id).squeeze()
        fld_team = self.team_emb(fld_team_id).squeeze()
        base1_run = self.bat_emb(base1_run_id).squeeze()
        base2_run = self.bat_emb(base2_run_id).squeeze()
        base3_run = self.bat_emb(base3_run_id).squeeze()
        away_start_bats = self.bat_emb(away_start_bats_ids).reshape(-1, 9 * self.emb_dim)
        home_start_bats = self.bat_emb(home_start_bats_ids).reshape(-1, 9 * self.emb_dim)
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

        values = self.big_in(values)

        values = self.shared(values)

        event_runs_ct = self.event_runs_ct(values)
        event_outs_ct = self.event_outs_ct(values)
        bat_dest = self.bat_dest(values)
        run1_dest = self.run1_dest(values)
        run2_dest = self.run2_dest(values)
        run3_dest = self.run3_dest(values)
        value_away = self.value_away(values)
        value_home = self.value_home(values)

        return event_runs_ct, event_outs_ct, \
            bat_dest, run1_dest, run2_dest, run3_dest, \
            value_away, value_home
