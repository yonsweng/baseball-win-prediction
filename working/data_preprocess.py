import pandas as pd

data = pd.read_csv('../input/mlbplaybyplay2010s/all2018.csv', low_memory=False)

# Leave only the PA data
data = data[data['BAT_EVENT_FL'] == 'T'].reset_index(drop=True)
pa_events = data['EVENT_CD'].unique()

data['HOME_TEAM_ID'] = data.apply(lambda row: row.GAME_ID[:3], axis=1)

# START_BASES_CD, END_BASES_CD = 4 * (3루) + 2 * (2루) + 1 * (1루)
data['START_BASES_CD'] = 1 * data['BASE1_RUN_ID'].notna() + \
                         2 * data['BASE2_RUN_ID'].notna() + \
                         4 * data['BASE3_RUN_ID'].notna()

data['END_BASES_CD'] = 1 * (data[['BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID', 'RUN3_DEST_ID']] == 1).apply(
    lambda x: sum(x) >= 1, axis=1) + \
                       2 * (data[['BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID', 'RUN3_DEST_ID']] == 2).apply(
    lambda x: sum(x) >= 1, axis=1) + \
                       4 * (data[['BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID', 'RUN3_DEST_ID']] == 3).apply(
    lambda x: sum(x) >= 1, axis=1)

data['EVENT_RUNS_CT'] = (data[['BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID', 'RUN3_DEST_ID']] >= 4).apply(
    lambda x: sum(x), axis=1)

data['END_OUTS_CT'] = data['OUTS_CT'] + data['EVENT_OUTS_CT']

# INN_NEW_FL: OUTS_CT == 0 and 주자 == 0, INN_END_FL: END_OUTS_CT == 3
data['INN_NEW_FL'] = (data['OUTS_CT'] == 0) & (data['START_BASES_CD'] == 0)
data['INN_END_FL'] = data['END_OUTS_CT'] == 3

# 초반, 후반 데이터 나누기.
n_games = sum(data.GAME_NEW_FL == 'T')  # 게임 수
train = data[data.GAME_ID.isin(data.GAME_ID.unique()[:n_games//2])]  # 전반기
test = data[data.GAME_ID.isin(data.GAME_ID.unique()[n_games//2:])]  # 후반기

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
