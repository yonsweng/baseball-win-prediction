import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_bats, num_pits, num_teams, emb_dim, dropout, device):
        super().__init__()

        # Embeddings
        self.bat_emb = nn.Embedding(num_bats, emb_dim)
        self.pit_emb = nn.Embedding(num_pits, emb_dim)
        self.team_emb = nn.Embedding(num_teams, emb_dim)

        # input layer
        self.input = nn.Sequential(
            nn.Linear(7 + 28 * emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Shared layers
        self.dense = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Policy output
        self.bat_dest = nn.Linear(256, 5)
        self.run1_dest = nn.Linear(256, 5)
        self.run2_dest = nn.Linear(256, 5)
        self.run3_dest = nn.Linear(256, 5)

        # Pred output
        self.pred_out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Memories
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_preds = []

        # Masks
        self.bat_mask = torch.tensor([0., 0., 0., 0., 0.], device=device)
        self.run1_mask = torch.tensor([0., 0., 0., 0., 0.], device=device)
        self.run2_mask = torch.tensor([0., -999., 0., 0., 0.], device=device)
        self.run3_mask = torch.tensor([0., -999., -999., 0., 0.], device=device)

    def forward(
        self,
        outs_ct,
        bat_id,
        pit_id,
        fld_team_id,
        base1_run_id,
        base2_run_id,
        base3_run_id,
        away_score_ct,
        home_score_ct,
        inn_ct,
        bat_home_id,
        away_bat_lineup,
        home_bat_lineup,
        away_start_bat_ids,
        home_start_bat_ids,
        away_pit_id,
        home_pit_id,
        away_team_id,
        home_team_id,
        softmax=True
    ):
        '''
        Returns:
            bat_dest  (BATCH_SIZE, 5),
            run1_dest (BATCH_SIZE, 5),
            run2_dest (BATCH_SIZE, 5),
            run3_dest (BATCH_SIZE, 5)
        '''
        # For policy and pred
        bat = self.bat_emb(bat_id).reshape(bat_id.shape[0], -1)
        pit = self.pit_emb(pit_id).reshape(pit_id.shape[0], -1)
        fld_team = self.team_emb(fld_team_id).reshape(fld_team_id.shape[0], -1)
        base1_run = self.bat_emb(base1_run_id).reshape(base1_run_id.shape[0], -1)
        base2_run = self.bat_emb(base2_run_id).reshape(base2_run_id.shape[0], -1)
        base3_run = self.bat_emb(base3_run_id).reshape(base3_run_id.shape[0], -1)

        # For pred
        away_start_bats = self.bat_emb(away_start_bat_ids).reshape(away_start_bat_ids.shape[0], -1)
        home_start_bats = self.bat_emb(home_start_bat_ids).reshape(home_start_bat_ids.shape[0], -1)
        away_start_pit = self.pit_emb(away_pit_id).reshape(away_pit_id.shape[0], -1)
        home_start_pit = self.pit_emb(home_pit_id).reshape(home_pit_id.shape[0], -1)
        away_team = self.team_emb(away_team_id).reshape(away_team_id.shape[0], -1)
        home_team = self.team_emb(home_team_id).reshape(home_team_id.shape[0], -1)

        x = torch.cat([
            outs_ct,
            bat,
            pit,
            fld_team,
            base1_run,
            base2_run,
            base3_run,
            away_score_ct,
            home_score_ct,
            inn_ct,
            bat_home_id,
            away_bat_lineup,
            home_bat_lineup,
            away_start_bats,
            home_start_bats,
            away_start_pit,
            home_start_pit,
            away_team,
            home_team
        ], dim=1)

        x = self.input(x)
        x = self.dense(x)
        bat_dest = self.bat_dest(x)
        run1_dest = self.run1_dest(x)
        run2_dest = self.run2_dest(x)
        run3_dest = self.run3_dest(x)
        bat_dest = self.bat_mask + bat_dest
        run1_dest = self.run1_mask + run1_dest
        run2_dest = self.run2_mask + run2_dest
        run3_dest = self.run3_mask + run3_dest

        if softmax:
            bat_dest = F.softmax(bat_dest, dim=1)
            run1_dest = F.softmax(run1_dest, dim=1)
            run2_dest = F.softmax(run2_dest, dim=1)
            run3_dest = F.softmax(run3_dest, dim=1)

        return bat_dest, run1_dest, run2_dest, run3_dest

    def v(self,
        outs_ct,
        bat_id,
        pit_id,
        fld_team_id,
        base1_run_id,
        base2_run_id,
        base3_run_id,
        away_score_ct,
        home_score_ct,
        inn_ct,
        bat_home_id,
        away_bat_lineup,
        home_bat_lineup,
        away_start_bat_ids,
        home_start_bat_ids,
        away_pit_id,
        home_pit_id,
        away_team_id,
        home_team_id
    ):
        '''
        Returns:
            pred      (BATCH_SIZE, 1)
        '''
        # For policy and pred
        bat = self.bat_emb(bat_id).reshape(bat_id.shape[0], -1)
        pit = self.pit_emb(pit_id).reshape(pit_id.shape[0], -1)
        fld_team = self.team_emb(fld_team_id).reshape(fld_team_id.shape[0], -1)
        base1_run = self.bat_emb(base1_run_id).reshape(base1_run_id.shape[0], -1)
        base2_run = self.bat_emb(base2_run_id).reshape(base2_run_id.shape[0], -1)
        base3_run = self.bat_emb(base3_run_id).reshape(base3_run_id.shape[0], -1)

        # For pred
        away_start_bats = self.bat_emb(away_start_bat_ids).reshape(away_start_bat_ids.shape[0], -1)
        home_start_bats = self.bat_emb(home_start_bat_ids).reshape(home_start_bat_ids.shape[0], -1)
        away_start_pit = self.pit_emb(away_pit_id).reshape(away_pit_id.shape[0], -1)
        home_start_pit = self.pit_emb(home_pit_id).reshape(home_pit_id.shape[0], -1)
        away_team = self.team_emb(away_team_id).reshape(away_team_id.shape[0], -1)
        home_team = self.team_emb(home_team_id).reshape(home_team_id.shape[0], -1)

        x = torch.cat([
            outs_ct,
            bat,
            pit,
            fld_team,
            base1_run,
            base2_run,
            base3_run,
            away_score_ct,
            home_score_ct,
            inn_ct,
            bat_home_id,
            away_bat_lineup,
            home_bat_lineup,
            away_start_bats,
            home_start_bats,
            away_start_pit,
            home_start_pit,
            away_team,
            home_team
        ], dim=1)

        x = self.input(x)
        x = self.dense(x)
        pred = self.pred_out(x)

        return pred
