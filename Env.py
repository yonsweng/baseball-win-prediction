class Env:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_state(self):
        return self.state

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
        self.state = state.copy()
        return self.state

    def step(self, action):
        dests = self.action_space.to_dests(action)

        runs_scored = [0, 0]  # [away, home]
        runs_scored[self.state['BAT_HOME_ID']] = self.get_runs_scored(dests)

        self.state['OUTS_CT'] += self.get_event_outs_ct(dests)

        if self.check_done(self.state) is not None:
            done = True
        else:
            done = False
            if self.state['OUTS_CT'] >= 3:
                self.change_batter()
                self.switch_field()
            else:
                self.move_runners(dests)
                self.change_batter()

        return self.state, tuple(runs_scored), done, None

    def get_runs_scored(self, dests):
        runs_scored = sum([dest >= 4 for dest in dests])

        if self.state['BAT_HOME_ID'] == 0:
            self.state['AWAY_SCORE_CT'] += runs_scored
        else:
            self.state['HOME_SCORE_CT'] += runs_scored

        return runs_scored

    def get_event_outs_ct(self, dests):
        event_outs_ct = int(dests[0] == 0)
        for i, dest in enumerate(dests[1:], 1):
            event_outs_ct += int(self.state[f'BASE{i}_RUN_ID'] != 0 and
                                 dest == 0)
        return event_outs_ct

    def check_done(self, state):
        done = False
        if state['INN_CT'] >= 9 and state['BAT_HOME_ID'] == 0 and\
            state['OUTS_CT'] >= 3 and\
                state['AWAY_SCORE_CT'] < state['HOME_SCORE_CT']:
            done = True  # n(>=9)회초 3아웃 홈팀 이기는 중
        elif state['INN_CT'] >= 9 and state['BAT_HOME_ID'] == 1 and\
                state['AWAY_SCORE_CT'] < state['HOME_SCORE_CT']:
            done = True  # n(>=9)회말 홈팀 이기는 중
        elif state['INN_CT'] >= 9 and state['BAT_HOME_ID'] == 1 and\
            state['OUTS_CT'] >= 3 and\
                state['AWAY_SCORE_CT'] > state['HOME_SCORE_CT']:
            done = True  # n(>=9)회말 3아웃 원정팀 이기는 중
        elif state['INN_CT'] >= 12 and state['BAT_HOME_ID'] == 1 and\
                state['OUTS_CT'] == 3:
            done = True  # 12회말 3아웃

        if done:
            return - (state['AWAY_END_SCORE_CT']
                      - state['AWAY_SCORE_CT']) ** 2 \
                   - (state['HOME_END_SCORE_CT']
                      - state['HOME_SCORE_CT']) ** 2
        else:
            return None

    def change_batter(self):
        if self.state['BAT_HOME_ID'] == 0:
            self.state['AWAY_BAT_LINEUP_ID'] = \
                self.state['AWAY_BAT_LINEUP_ID'] % 9 + 1
        else:
            self.state['HOME_BAT_LINEUP_ID'] = \
                self.state['HOME_BAT_LINEUP_ID'] % 9 + 1

    def switch_field(self):
        for i in range(1, 4):
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

        if dests[0] != 0 and dests[0] != 4:
            self.state[f'BASE{dests[0]}_RUN_ID'] = batter

        # move the other runners
        for i, dest in enumerate(dests[1:], 1):
            if dest != 0 and dest != 4:
                self.state[f'BASE{dests[i]}_RUN_ID'] = \
                    self.state[f'BASE{i}_RUN_ID']
