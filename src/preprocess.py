''' Preprocess '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

diff_cols = [
    'INN_CT',
    'BAT_HOME_ID',
    'OUTS_CT',
    'BASE1_RUN_ID',
    'BASE2_RUN_ID',
    'BASE3_RUN_ID',
    'BAT_LINEUP_ID',
    'PIT_LINEUP_ID',
    'AWAY_SCORE_CT',
    'HOME_SCORE_CT'
]
used_cols = [
    'GAME_ID',
    'AWAY_TEAM_ID',
    'HOME_TEAM_ID',
    'GAME_NEW_FL',
    'GAME_END_FL',
    'AWAY_START_PIT_ID',
    'HOME_START_PIT_ID',
    'INN_END_FL',
    'FINAL_AWAY_SCORE_CT',
    'FINAL_HOME_SCORE_CT',
] + [f'AWAY_START_BAT{i}_ID' for i in range(1, 10)] \
  + [f'HOME_START_BAT{i}_ID' for i in range(1, 10)] \
  + diff_cols \
  + ['END_' + col for col in diff_cols]


def preprocess(data):
    '''
    Returns:
        data
    '''
    # Use only PA data.
    data = data[data['BAT_EVENT_FL'] == 'T'].reset_index(drop=True)

    data['INN_END_FL'] = (data['OUTS_CT'] + data['EVENT_OUTS_CT'] == 3) \
                         | (data['GAME_END_FL'] == 'T')

    # Encode teams.
    data['HOME_TEAM_ID'] = data['GAME_ID'].apply(lambda x: x[:3])
    team_encoder = OrdinalEncoder()
    team_encoder.categories_ = \
        np.unique(data[['AWAY_TEAM_ID', 'HOME_TEAM_ID']].values.reshape(-1)).reshape(1, -1)
    data['AWAY_TEAM_ID'] = \
        team_encoder.transform(data['AWAY_TEAM_ID'].values.reshape(-1, 1)).astype(int)
    data['HOME_TEAM_ID'] = \
        team_encoder.transform(data['HOME_TEAM_ID'].values.reshape(-1, 1)).astype(int)

    # Encode batters.
    data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']] = \
        data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']].fillna('')
    bat_encoder = OrdinalEncoder()
    bat_encoder.categories_ = \
        np.unique(data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']].values
            .reshape(-1)).reshape(1, -1)
    data['BAT_ID'] = bat_encoder.transform(data['BAT_ID'].values.reshape(-1, 1)).astype(int)
    data['BASE1_RUN_ID'] = bat_encoder.transform(data['BASE1_RUN_ID'].values.reshape(-1, 1))\
        .astype(int)
    data['BASE2_RUN_ID'] = bat_encoder.transform(data['BASE2_RUN_ID'].values.reshape(-1, 1))\
        .astype(int)
    data['BASE3_RUN_ID'] = bat_encoder.transform(data['BASE3_RUN_ID'].values.reshape(-1, 1))\
        .astype(int)

    # Encode pitchers.
    pit_encoder = OrdinalEncoder()
    data['PIT_ID'] = pit_encoder.fit_transform(data['PIT_ID'].values.reshape(-1, 1)).astype(int)

    # Runs scored on the events.
    data['EVENT_RUNS_CT'] = \
        (data[['BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID', 'RUN3_DEST_ID']] >= 4).apply(sum,
        axis=1)

    # Save starting batters and pitchers' IDs of games.
    games = [game for _, game in data.groupby(data['GAME_ID'])]
    new_games = []

    for game in games:
        away_bats = game['BAT_ID'][game['BAT_HOME_ID'] == 0].values[:9]
        home_bats = game['BAT_ID'][game['BAT_HOME_ID'] == 1].values[:9]
        away_pit = game['PIT_ID'][game['BAT_HOME_ID'] == 0].values[0]
        home_pit = game['PIT_ID'][game['BAT_HOME_ID'] == 1].values[0]

        for idx in range(9):
            game[f'AWAY_START_BAT{idx + 1}_ID'] = away_bats[idx]
        for idx in range(9):
            game[f'HOME_START_BAT{idx + 1}_ID'] = home_bats[idx]

        game['AWAY_START_PIT_ID'] = away_pit
        game['HOME_START_PIT_ID'] = home_pit
        game['PIT_LINEUP_ID'] = \
            ((game['PIT_ID'] != game['AWAY_START_PIT_ID']) & \
            (game['PIT_ID'] != game['HOME_START_PIT_ID'])).astype(int)

        last_pa = game.iloc[-1:][diff_cols + ['EVENT_RUNS_CT']]
        if last_pa.iloc[0]['BAT_HOME_ID'] == 0:
            last_pa.iloc[0]['AWAY_SCORE_CT'] += last_pa.iloc[0]['EVENT_RUNS_CT']
        else:
            last_pa.iloc[0]['HOME_SCORE_CT'] += last_pa.iloc[0]['EVENT_RUNS_CT']

        end_df = pd.concat([game.iloc[1:][diff_cols + ['EVENT_RUNS_CT']], last_pa],
            ignore_index=True)
        end_df.rename(columns={col: 'END_' + col for col in diff_cols}, inplace=True)
        game.reset_index(inplace=True, drop=True)
        end_df.reset_index(inplace=True, drop=True)

        new_game = pd.concat([game, end_df], axis=1)

        # Get scores when the game ends.
        new_game['FINAL_AWAY_SCORE_CT'] = new_game.iloc[-1]['END_AWAY_SCORE_CT']
        new_game['FINAL_HOME_SCORE_CT'] = new_game.iloc[-1]['END_HOME_SCORE_CT']

        new_games.append(new_game)

    data = pd.concat(new_games, ignore_index=False)
    data = data[used_cols]

    # Transform types of the data.
    data['GAME_END_FL'] = data['GAME_END_FL'] == 'T'

    print('Data preprocess done.')
    return data
