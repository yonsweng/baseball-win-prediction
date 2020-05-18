import pandas as pd


class Env:
    def __init__(self, train):
        self.ob = {}
        self.inn_end_score = 0
        self.away_batters = None
        self.home_batters = None
        self.away_pitcher = None
        self.home_pitcher = None
        self.away_batter_order = 0  # 0 ~ 8
        self.home_batter_order = 0  # 0 ~ 8

        # probability of (state, action) -> next_state
        end_state_cnt = pd.pivot_table(train, values='EVENT_ID', index=['OUTS_CT', 'START_BASES_CD', 'EVENT_CD'],
                                       columns=['EVENT_RUNS_CT', 'END_OUTS_CT', 'END_BASES_CD'],
                                       aggfunc='count').fillna(0.0).astype(int)
        state_action_cnt = train.groupby(['OUTS_CT', 'START_BASES_CD'])['EVENT_CD'].value_counts()
        self.sa_s = end_state_cnt.apply(lambda x: x / state_action_cnt[end_state_cnt.index])

    def reset(self, this_game):
        self.this_inn = this_inn

        # get starting batters and pitcher
        self.away_batters = list(self.this_game['BAT_ID'][self.this_game['BAT_HOME_ID'] == 0].head(9))
        self.home_batters = list(self.this_game['BAT_ID'][self.this_game['BAT_HOME_ID'] == 1].head(9))
        self.home_pitcher = self.this_game['PIT_ID'][self.this_game['BAT_HOME_ID'] == 0].iloc[0]
        self.away_pitcher = self.this_game['PIT_ID'][self.this_game['BAT_HOME_ID'] == 1].iloc[0]

        # get inn_end_score
        tmp = self.this_game[['INN_CT', 'BAT_HOME_ID', 'AWAY_SCORE_CT', 'HOME_SCORE_CT']][self.this_game['INN_END_FL']]
        for idx, row in tmp.iterrows():
            self.inn_end_score[row['INN_CT'], row['BAT_HOME_ID']] = \
                row['AWAY_SCORE_CT'] if row['BAT_HOME_ID'] == 0 else row['HOME_SCORE_CT']

        # reset batting order
        self.away_batter_order = 0  # 0 ~ 8
        self.home_batter_order = 0  # 0 ~ 8

        # reset observation
        self.ob['AWAY_SCORE_CT'] = 0
        self.ob['HOME_SCORE_CT'] = 0
        self.ob['INN_CT'] = 1
        self.ob['BAT_HOME_ID'] = 0
        self.ob['OUTS_CT'] = 0
        self.ob['BASES_CD'] = 0
        self.ob['INN_END_FL'] = False
        self.ob['BAT_ID'] = self.away_batters[self.away_batter_order]
        self.ob['PIT_ID'] = self.home_pitcher

        return self.ob

    def step(self, action):
        rwd = 0
        done = False
        info = None

        # rotate batting order
        if self.ob['BAT_HOME_ID'] == 0:
            self.away_batter_order = (self.away_batter_order + 1) % 9
        else:
            self.home_batter_order = (self.home_batter_order + 1) % 9

        # next state
        pr = self.sa_s.loc[self.ob['OUTS_CT'], self.ob['BASES_CD'], action]
        tmp = pr.sample(weights=pr).index[0]

        # update score
        if self.ob['BAT_HOME_ID'] == 0:  # away team batting
            self.ob['AWAY_SCORE_CT'] += tmp[0]
        else:  # home team batting
            self.ob['HOME_SCORE_CT'] += tmp[0]

        # update outs and bases
        self.ob['OUTS_CT'], self.ob['BASES_CD'] = tmp[1], tmp[2]

        if self.ob['OUTS_CT'] == 3:
            # calculate reward
            score = self.ob['AWAY_SCORE_CT'] if self.ob['BAT_HOME_ID'] == 0 else self.ob['HOME_SCORE_CT']
            rwd = -abs(score - self.inn_end_score[self.ob['INN_CT'], self.ob['BAT_HOME_ID']]) \
                if (self.ob['INN_CT'], self.ob['BAT_HOME_ID']) in self.inn_end_score else 0

            self.ob['INN_CT'] += 1 if self.ob['BAT_HOME_ID'] == 1 else 0
            self.ob['BAT_HOME_ID'] = 1 - self.ob['BAT_HOME_ID']
            self.ob['END_OUTS_CT'] = 0
            self.ob['BASES_CD'] = 0
            self.ob['INN_END_FL'] = True
            self.ob['OUTS_CT'] = 0
        else:
            self.ob['INN_END_FL'] = False

        if self.ob['INN_CT'] > 9:
            done = True

        return self.ob, rwd, done, info


def make(train):
    return Env(train)
