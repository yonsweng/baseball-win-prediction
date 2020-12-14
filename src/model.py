import torch
import torch.nn as nn


class Model(nn.Module):
    '''
    Baseball model
    '''
    def __init__(self, num_bats, num_pits, num_teams, embedding_dim):
        super().__init__()

        # Layer parameters
        self.embedding_dim = embedding_dim
        hidden_dim = 64
        small_embedding_dim = 8

        # Embedding layers
        self.bat_emb = nn.Embedding(num_bats, self.embedding_dim, 0)
        self.pit_emb = nn.Embedding(num_pits, self.embedding_dim)
        self.team_emb = nn.Embedding(num_teams, self.embedding_dim)

        # State layers
        self.base_run_model = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, small_embedding_dim)
        )
        self.state_model = nn.Sequential(
            nn.Linear(small_embedding_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )

        # Event model
        self.event_model = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.reward_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.inn_end_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.next_state_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )

        # Value model
        self.bat_model = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )
        self.value_model = nn.Sequential(
            nn.Linear(4 * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_game_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.value_away_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_home_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_state(self,
            away_score_ct,
            home_score_ct,
            inn_ct,
            bat_home_id,
            pit_lineup,
            outs_ct,
            base1_run_id,
            base2_run_id,
            base3_run_id):
        '''
        Returns:
            state
        '''
        base1_run = self.bat_emb(base1_run_id)
        base2_run = self.bat_emb(base2_run_id)
        base3_run = self.bat_emb(base3_run_id)

        # base_run_in = (batch_size, 3, embedding_dim)
        base_run_in = torch.cat([base1_run, base2_run, base3_run], dim=1)
        base_run = self.base_run_model(base_run_in)

        state_in = torch.cat([away_score_ct,
                              home_score_ct,
                              inn_ct,
                              bat_home_id,
                              pit_lineup,
                              outs_ct,
                              base_run], dim=1)
        state = self.state_model(state_in)

        return state

    def forward(self,
            start_pit_id,
            fld_team_id,
            state,
            bat_this_id,
            bat_next_ids):
        '''
        Returns:
            next_state,
            reward,
            inn_end_fl,
            value_game,
            value_away,
            value_home
        '''
        # Get embeddings
        start_pit = self.pit_emb(start_pit_id)
        fld_team = self.team_emb(fld_team_id)
        bat_this = self.bat_emb(bat_this_id)
        bat_next = self.bat_emb(bat_next_ids)  # (batch_size, 8, embedding_dim)

        # Get an inn_end_fl, a reward and a next state
        event_in = torch.cat([start_pit.squeeze(),
                              fld_team.squeeze(),
                              state.squeeze(),
                              bat_this.squeeze()], dim=1)
        event = self.event_model(event_in)
        reward = self.reward_model(event)
        inn_end_fl = self.inn_end_model(event)
        next_state = self.next_state_model(event)

        # Get a next pit_lineup, value_game, value_away and value_home
        bat = self.bat_model(torch.cat([bat_this, bat_next], dim=1))  # (batch_size, embedding_dim)
        value_in = torch.cat([start_pit.squeeze(),
                              fld_team.squeeze(),
                              state.squeeze(),
                              bat.squeeze()], dim=1)
        value = self.value_model(value_in)
        value_game = self.value_game_model(value)
        value_away = self.value_away_model(value)
        value_home = self.value_home_model(value)

        return next_state, reward, inn_end_fl, value_game, value_away, value_home
