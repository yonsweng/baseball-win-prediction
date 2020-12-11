import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseballModel(nn.Module):
    def __init__(self, num_bats, num_pits, num_teams, embedding_dim):
        super().__init__()

    def representation(self):
        return

    def forward(self,
            state,
            start_pit,
            fld_team,
            pit_lineup,
            bat,
            bat_next1,
            bat_next2,
            bat_next3,
            bat_next4,
            bat_next5,
            bat_next6,
            bat_next7,
            bat_next8,
            bat_next9):
        pit = (1 - pit_lineup) * start_pit + pit_lineup * fld_team
        return pit

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
