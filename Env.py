import torch
from ActionSpace import ActionSpace


class Env:
    def __init__(self):
        self.action_space = ActionSpace()

    def get_obs(self):
        away_offset = self.state['AWAY_BAT_LINEUP_ID']
        home_offset = self.state['HOME_BAT_LINEUP_ID']
        return {
            'float': torch.tensor([
                self.state['INN_CT'] / 9,  # 1/9, 2/9, ..., 1, ...
                (self.state['BAT_HOME_ID'] - 0.5) * 2,  # -1, 1
                (self.state['OUTS_CT'] + 1) / 3  # 1/3, 2/3, 1
            ], dtype=torch.float),
            'bat': torch.tensor([
                *[self.state['BASE1_RUN_ID'],
                  self.state['BASE2_RUN_ID'],
                  self.state['BASE3_RUN_ID']],
                *[self.state[f'AWAY_BAT{(i+away_offset-1)%9+1}_ID']
                  for i in range(9)],
                *[self.state[f'HOME_BAT{(i+home_offset-1)%9+1}_ID']
                  for i in range(9)]
            ], dtype=torch.long),
            'pit': torch.tensor([
                self.state['AWAY_PIT_ID'],
                self.state['HOME_PIT_ID']
            ], dtype=torch.long),
            'team': torch.tensor([
                self.state['AWAY_TEAM_ID'],
                self.state['HOME_TEAM_ID']
            ], dtype=torch.long)
        }

    def reset(self, state):
        '''
        state:
            'AWAY_TEAM_ID',
            'HOME_TEAM_ID',
            'INN_CT',
            'BAT_HOME_ID',
            'OUTS_CT',
            'BASE1_RUN_ID',
            'BASE2_RUN_ID',
            'BASE3_RUN_ID',
            'BAT_DEST_ID',
            'RUN1_DEST_ID',
            'RUN2_DEST_ID',
            'RUN3_DEST_ID',
            'AWAY_SCORE_CT',
            'HOME_SCORE_CT',
            'AWAY_PIT_ID',
            'HOME_PIT_ID',
            'AWAY_BAT_LINEUP_ID',
            'HOME_BAT_LINEUP_ID',
            'AWAY_END_SCORE_CT',
            'HOME_END_SCORE_CT'
            'AWAY_BATi_ID',
            'HOME_BATi_ID'
        '''
        self.state = state
        return self.get_obs()

    def step(self, action):
        dests = self.action_space.to_dests(action)

        runs_scored = [0, 0]
        runs_scored[self.state['BAT_HOME_ID']] = self.get_runs_scored(dests)

        self.state['OUTS_CT'] += self.get_event_outs_ct(dests)

        if self.check_done():
            return self.get_obs(), tuple(runs_scored), True, None

        if self.state['OUTS_CT'] >= 3:
            self.change_batter()
            self.switch_field()
        else:
            self.move_runners(dests)
            self.change_batter()

        return self.get_obs(), tuple(runs_scored), False, None

    def get_runs_scored(self, dests):
        runs_scored = sum([dest >= 4 for dest in dests])

        if self.state['BAT_HOME_ID'] == 0:
            self.state['AWAY_SCORE_CT'] += runs_scored
        else:
            self.state['HOME_SCORE_CT'] += runs_scored

    def get_event_outs_ct(self, dests):
        event_outs_ct = int(dests[0] == 0)
        for i, dest in enumerate(dests[1:], 1):
            event_outs_ct += int(self.state[f'BASE{i}_RUN_ID'] != 0 and
                                 dest == 0)
        return event_outs_ct

    def check_done(self):
        if self.state['INN_CT'] >= 9 and self.state['BAT_HOME_ID'] == 0 and\
            self.state['OUTS_CT'] >= 3 and\
                self.state['AWAY_SCORE_CT'] < self.state['HOME_SCORE_CT']:
            return True  # n(>=9)회초 3아웃 홈팀 이기는 중
        elif self.state['INN_CT'] >= 9 and self.state['BAT_HOME_ID'] == 1 and\
                self.state['AWAY_SCORE_CT'] < self.state['HOME_SCORE_CT']:
            return True  # n(>=9)회말 홈팀 이기는 중
        elif self.state['INN_CT'] >= 9 and self.state['BAT_HOME_ID'] == 1 and\
            self.state['OUTS_CT'] >= 3 and\
                self.state['AWAY_SCORE_CT'] > self.state['HOME_SCORE_CT']:
            return True  # n(>=9)회말 3아웃 원정팀 이기는 중
        elif self.state['INN_CT'] >= 12 and self.state['BAT_HOME_ID'] == 1 and\
                self.state['OUTS_CT'] == 3:
            return True  # 12회말 3아웃

        return False

    def change_batter(self):
        if self.state['BAT_HOME_ID'] == 0:
            self.state['AWAY_BAT_LINEUP_ID'] = \
                self.state['AWAY_BAT_LINEUP_ID'] % 9 + 1
        else:
            self.state['HOME_BAT_LINEUP_ID'] = \
                self.state['HOME_BAT_LINEUP_ID'] % 9 + 1

    def switch_field(self):
        for i in range(3):
            self.state[f'BASE{i}_RUN_ID'] = 0
        self.state['OUTS_CT'] = 0
        self.state['INN_CT'] += self.state['BAT_HOME_ID']
        self.state['BAT_HOME_ID'] = 1 - self.state['BAT_HOME_ID']

    def move_runners(self, dests):
        # move the batter
        if self.state['BAT_HOME_ID'] == 0:
            batter = self.state['AWAY_BAT{}_ID'.format(
                self.state['AWAY_BAT_LINEUP_ID'])]
        else:
            batter = self.state['HOME_BAT{}_ID'.format(
                self.state['HOME_BAT_LINEUP_ID'])]
        self.state[f'BASE{dests[0]}_RUN_ID'] = batter

        # move the other runners
        for i, dest in enumerate(dests[1:], 1):
            if dest != 0 and dest != 4:
                self.state[f'BASE{dests[i]}_RUN_ID'] = \
                    self.state[f'BASE{i}_RUN_ID']
