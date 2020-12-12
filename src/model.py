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
        small_embedding_dim = 4

        # Embedding layers
        self.bat_emb = nn.Embedding(num_bats, self.embedding_dim, 0)
        self.pit_emb = nn.Embedding(num_pits, self.embedding_dim)
        self.team_emb = nn.Embedding(num_teams, self.embedding_dim)

        # State layers
        self.base_run_model = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, small_embedding_dim)
        )
        self.pa_state_model = nn.Sequential(
            nn.Linear(small_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )
        self.game_state_model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )

        # PA model
        self.pa_model = nn.Sequential(
            nn.Linear(5 * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.inn_end_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.reward_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.next_pa_state_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embedding_dim)
        )

        # Game model
        self.situation_model = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, small_embedding_dim)
        )
        self.bat_model = nn.Sequential(
            nn.Linear(9 * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, small_embedding_dim)
        )
        self.game_model = nn.Sequential(
            nn.Linear(2 * small_embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pit_lineup_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_game = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_away = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_home = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_pa_state(self,
            outs_ct,
            base1_run_id,
            base2_run_id,
            base3_run_id):
        '''
        Returns:
            pa_state
        '''
        base1_run = self.bat_emb(base1_run_id)
        base2_run = self.bat_emb(base2_run_id)
        base3_run = self.bat_emb(base3_run_id)

        base_run_in = torch.cat([base1_run, base2_run, base3_run], dim=1)
        base_run_out = self.base_run_model(base_run_in)

        pa_state_input = torch.cat([outs_ct, base_run_out], dim=1)
        pa_state = self.pa_state_model(pa_state_input)

        return pa_state

    def forward(self,
            start_pit_id,
            fld_team_id,
            pa_state,
            away_score_ct,
            home_score_ct,
            inn_ct,
            bat_home_id,
            pit_lineup,
            bat_this_id,
            bat_next_ids):
        '''
        Returns:
            inn_end_fl,
            reward,
            next_pa_state,
            pit_lineup,
            value_game,
            value_away,
            value_home
        '''
        # Get a game state
        game_state_in = torch.cat([away_score_ct.unsqueeze(1),
                                   home_score_ct.unsqueeze(1),
                                   inn_ct.unsqueeze(1),
                                   bat_home_id.unsqueeze(1)], dim=1)
        game_state = self.game_state_model(game_state_in)

        # Get a pitcher's embedding
        start_pit = self.pit_emb(start_pit_id)
        fld_team = self.team_emb(fld_team_id)
        pit = (1 - pit_lineup) * start_pit + pit_lineup * fld_team

        # Get an inn_end_fl, a reward and a PA state
        pa_in = torch.cat([pit, fld_team, pa_state, game_state], dim=1)
        pa = self.pa_model(pa_in)
        inn_end_fl = self.inn_end_model(pa)
        reward = self.reward_model(pa)
        next_pa_state = self.next_pa_state_model(pa)

        # Get a batters' embedding
        bat_this = self.bat_emb(bat_this_id)
        bat_next = self.bat_emb(bat_next_ids)  # (batch_size, 8, embedding_dim)
        bat_in = torch.cat([bat_this.unsqueeze(1), bat_next], dim=1).reshape(-1, 9 * self.embedding_dim)
        bat = self.bat_model(bat_in)

        # Situation embedding
        situation_in = torch.cat([start_pit, fld_team, pa_state, game_state], dim=1)
        situation = self.situation_model(situation_in)

        # Get a next pit_lineup, value_game, value_away and value_home
        game_in = torch.cat([situation, bat, pit_lineup.unsqueeze(1)], dim=1)
        game = self.game_model(game_in)
        pit_lineup = self.pit_lineup_model(game)
        value_game = self.value_game_model(game)
        value_away = self.value_away_model(game)
        value_home = self.value_home_model(game)

        return inn_end_fl, reward, next_pa_state, pit_lineup, value_game, value_away, value_home

    #     self.representation_layers = [
    #         nn.Linear(7 * embedding_dim + 5, hidden_dim), nn.ReLU()]
    #     for _ in range(num_hidden_layers):
    #         self.representation_layers += [
    #             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    #     self.representation_layers.append(nn.Linear(hidden_dim, embedding_dim))

    #     self.dynamics_layers = [
    #         nn.Linear(embedding_dim, hidden_dim), nn.ReLU()]
    #     for _ in range(num_hidden_layers):
    #         self.dynamics_layers += [nn.Linear(hidden_dim,
    #                                            hidden_dim), nn.ReLU()]
    #     self.dynamics_layers.append(nn.Linear(hidden_dim, embedding_dim + 3))

    #     self.value_layers = [
    #         nn.Linear(embedding_dim, hidden_dim), nn.ReLU()]
    #     for _ in range(num_hidden_layers):
    #         self.value_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    #     self.value_layers.append(nn.Linear(hidden_dim, 2))

    # # AWAY_START_BATS_IDS, HOME_START_BATS_IDS = torch.long(BATCH_SIZE, 9)
    # # BAT_HOME_ID = torch.bool(BATCH_SIZE, )
    # # INN_CT, OUTS_CT, AWAY_SCORE_CT, HOME_SCORE_CT = torch.float(BATCH_SIZE, )
    # def representation(self,
    #                    AWAY_TEAM_ID,
    #                    HOME_TEAM_ID,
    #                    AWAY_START_BATS_IDS,
    #                    HOME_START_BATS_IDS,
    #                    AWAY_START_PIT_ID,
    #                    HOME_START_PIT_ID,
    #                    BASE1_RUN_ID,
    #                    BASE2_RUN_ID,
    #                    BASE3_RUN_ID,
    #                    BAT_LINEUP_ID,
    #                    PIT_LINEUP_ID,  # torch.bool(BATCH_SIZE, ). 0이면 선발, 1이면 계투
    #                    BAT_HOME_ID,
    #                    INN_CT,
    #                    OUTS_CT,
    #                    AWAY_SCORE_CT,
    #                    HOME_SCORE_CT):
    #     embedding_away_team = self.embedding_team(AWAY_TEAM_ID)
    #     embedding_home_team = self.embedding_team(HOME_TEAM_ID)

    #     start_bats_ids = torch.where(BAT_HOME_ID.unsqueeze(1).repeat(
    #         1, 9), HOME_START_BATS_IDS, AWAY_START_BATS_IDS)
    #     embedding_bat = self.embedding_bat(
    #         (start_bats_ids * torch.eye(9)[BAT_LINEUP_ID - 1]).sum(1).type(torch.long))

    #     start_pit_id = torch.where(
    #         BAT_HOME_ID, HOME_START_PIT_ID, AWAY_START_PIT_ID)
    #     embedding_pit = self.embedding_pit(start_pit_id)

    #     embedding_base1 = self.embedding_bat(BASE1_RUN_ID)
    #     embedding_base2 = self.embedding_bat(BASE2_RUN_ID)
    #     embedding_base3 = self.embedding_bat(BASE3_RUN_ID)

    #     x = torch.cat((embedding_away_team,
    #                    embedding_home_team,
    #                    embedding_bat,
    #                    embedding_pit,
    #                    embedding_base1,
    #                    embedding_base2,
    #                    embedding_base3,
    #                    BAT_HOME_ID.unsqueeze(1),
    #                    INN_CT.unsqueeze(1),
    #                    OUTS_CT.unsqueeze(1),
    #                    AWAY_SCORE_CT.unsqueeze(1),
    #                    HOME_SCORE_CT.unsqueeze(1)), 1)

    #     for layer in self.representation_layers:
    #         x = layer(x)
    #     return x  # (BATCH_SIZE, embedding_dim)

    # '''
    # Args:
    #     state: torch.Tensor(BATCH_SIZE, embedding_dim)
    # Return:
    #     state, reward_away, reward_home, done
    # '''
    # def dynamics(self, state):
    #     x = state
    #     for layer in self.dynamics_layers:
    #         x = layer(x)
    #     state, reward_away, reward_home, done = x[:,
    #                                               :-3], x[:, -3], x[:, -2], x[:, -1]
    #     return state, reward_away, reward_home, done

    # def value(self, state):
    #     x = state
    #     for layer in self.value_layers:
    #         x = layer(x)
    #     value_away, value_home = x[:, 0], x[:, 1]
    #     return value_away, value_home  # (BATCH_SIZE, ), (BATCH_SIZE, )
