import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseballModel(nn.Module):
    def __init__(self, num_pits, num_teams, num_bats, embedding_dim, hidden_dim, num_hidden_layers):
        super().__init__()
        self.embedding_pit = nn.Embedding(num_pits, embedding_dim)
        self.embedding_team = nn.Embedding(num_teams, embedding_dim)
        self.embedding_bat = nn.Embedding(num_bats, embedding_dim)

        self.representation_layers = [
            nn.Linear(7 * embedding_dim + 4, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            self.representation_layers += [
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.representation_layers.append(nn.Linear(hidden_dim, embedding_dim))

        self.dynamics_layers = [
            nn.Linear(2 * embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            self.dynamics_layers += [nn.Linear(hidden_dim,
                                               hidden_dim), nn.ReLU()]
        self.dynamics_layers.append(nn.Linear(hidden_dim, embedding_dim + 2))

        self.value_layers = [
            nn.Linear(2 * embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            self.value_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.value_layers.append(nn.Linear(hidden_dim, 1))

    # START_BATS_IDS = torch.long(BATCH_SIZE, 9)
    # START_PIT_ID, FLD_TEAM_ID, BAT_LINEUP_ID, BASE1_RUN_ID, BASE2_RUN_ID, BASE3_RUN_ID = torch.long(BATCH_SIZE, )
    # OUTS_CT, INN_CT, START_BAT_SCORE_CT = torch.float(BATCH_SIZE, )
    def representation(self, START_PIT_ID, FLD_TEAM_ID, BAT_LINEUP_ID, START_BATS_IDS, BASE1_RUN_ID, BASE2_RUN_ID, BASE3_RUN_ID, OUTS_CT, INN_CT, START_FLD_SCORE_CT, START_BAT_SCORE_CT):
        embedding_start_pit = self.embedding_pit(START_PIT_ID)
        embedding_fld_team = self.embedding_team(FLD_TEAM_ID)
        embedding_bat = self.embedding_bat(
            (START_BATS_IDS * torch.eye(9)[BAT_LINEUP_ID - 1]).sum(1).type(torch.long))
        embedding_start_bats = self.embedding_bat(START_BATS_IDS).sum(1)
        embedding_base1 = self.embedding_bat(BASE1_RUN_ID)
        embedding_base2 = self.embedding_bat(BASE2_RUN_ID)
        embedding_base3 = self.embedding_bat(BASE3_RUN_ID)
        x = torch.cat((embedding_start_pit, embedding_fld_team, embedding_bat, embedding_start_bats, embedding_base1,
                       embedding_base2, embedding_base3, OUTS_CT.unsqueeze(1), INN_CT.unsqueeze(1), START_FLD_SCORE_CT.unsqueeze(1), START_BAT_SCORE_CT.unsqueeze(1)), 1)
        for layer in self.representation_layers:
            x = layer(x)
        return x  # (BATCH_SIZE, embedding_dim)

    # BAT_LINEUP_ID = (BATCH_SIZE, )
    # state = (BATCH_SIZE, embedding_dim)
    def dynamics(self, BAT_LINEUP_ID, START_BATS_IDS, state):
        embedding_bat = self.embedding_bat(
            (START_BATS_IDS * torch.eye(9)[BAT_LINEUP_ID - 1]).sum(1).type(torch.long))
        x = torch.cat((embedding_bat, state), 1)
        for layer in self.dynamics_layers:
            x = layer(x)
        state, reward, done = x[:, :-2], x[:, -2], x[:, -1]
        return BAT_LINEUP_ID % 9 + 1, state, reward, done

    def value(self, BAT_LINEUP_ID, START_BATS_IDS, state):
        embedding_bat = self.embedding_bat(
            (START_BATS_IDS * torch.eye(9)[BAT_LINEUP_ID - 1]).sum(1).type(torch.long))
        x = torch.cat((embedding_bat, state), 1)
        for layer in self.value_layers:
            x = layer(x)
        return x.squeeze()  # (BATCH_SIZE, )
