import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

used_cols = [
    'GAME_ID',
    'AWAY_START_PIT_ID',
    'HOME_START_PIT_ID',
    'AWAY_TEAM_ID',
    'HOME_TEAM_ID',
    'AWAY_BAT_LINEUP_ID',
    'HOME_BAT_LINEUP_ID',
    'AWAY_SCORE_CT',
    'HOME_SCORE_CT',
    'INN_CT',
    'BAT_HOME_ID',
    'OUTS_CT',
    'BAT_ID',
    'PIT_ID',
    'BASE1_RUN_ID',
    'BASE2_RUN_ID',
    'BASE3_RUN_ID',
    'EVENT_RUNS_CT',
    'EVENT_OUTS_CT',
    'BAT_DEST_ID',
    'RUN1_DEST_ID',
    'RUN2_DEST_ID',
    'RUN3_DEST_ID',
    'VALUE_AWAY',
    'VALUE_HOME',
    'GAME_NEW_FL'
] + [f'AWAY_START_BAT{i}_ID' for i in range(1, 10)] \
  + [f'HOME_START_BAT{i}_ID' for i in range(1, 10)]


def preprocess(data):
    '''
    Returns:
        data
    '''
    # Use only PA data.
    data = data[data['BAT_EVENT_FL'] == 'T'].reset_index(drop=True)

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

    for game in games:
        # AWAY_START_BATi_ID, HOME_START_BATi_ID
        away_bats = game['BAT_ID'][game['BAT_HOME_ID'] == 0].values[:9]
        home_bats = game['BAT_ID'][game['BAT_HOME_ID'] == 1].values[:9]
        for idx in range(9):
            game[f'AWAY_START_BAT{idx + 1}_ID'] = away_bats[idx]
        for idx in range(9):
            game[f'HOME_START_BAT{idx + 1}_ID'] = home_bats[idx]

        # AWAY_START_PIT_ID, HOME_START_PIT_ID
        away_pit = game['PIT_ID'][game['BAT_HOME_ID'] == 1].values[0]
        home_pit = game['PIT_ID'][game['BAT_HOME_ID'] == 0].values[0]
        game['AWAY_START_PIT_ID'] = away_pit
        game['HOME_START_PIT_ID'] = home_pit

        # AWAY_BAT_LINEUP_ID, HOME_BAT_LINEUP_ID
        bat_home_id = list(game['BAT_HOME_ID'].values)
        bat_lineup_id = list(game['BAT_LINEUP_ID'].values)
        away_bat_lineup = []
        home_bat_lineup = []
        for i, bat_home in enumerate(bat_home_id):
            if bat_home == 0:  # Away bat
                away_bat_lineup.append(bat_lineup_id[i])
                home_bat_lineup.append(home_bat_lineup[i-1] if i > 0 else 1)
            else:              # Home bat
                home_bat_lineup.append(bat_lineup_id[i])
                away_bat_lineup.append(away_bat_lineup[i-1] if i > 0 else 1)
        game['AWAY_BAT_LINEUP_ID'] = away_bat_lineup
        game['HOME_BAT_LINEUP_ID'] = home_bat_lineup

        # Get scores to get until end.
        last_pa = game.iloc[-1]
        last_away_score_ct = last_pa['AWAY_SCORE_CT'] + \
            (last_pa['EVENT_RUNS_CT'] if last_pa['BAT_HOME_ID'] == 0 else 0)
        last_home_score_ct = last_pa['HOME_SCORE_CT'] + \
            (last_pa['EVENT_RUNS_CT'] if last_pa['BAT_HOME_ID'] == 1 else 0)
        game['VALUE_AWAY'] = last_away_score_ct - game['AWAY_SCORE_CT'] 
        game['VALUE_HOME'] = last_home_score_ct - game['HOME_SCORE_CT']

    data = pd.concat(games, ignore_index=False)
    data = data[used_cols]
    print('Data preprocess done.')

    return data
