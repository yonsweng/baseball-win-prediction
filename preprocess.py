import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder


def leave_only_bat_events(data):
    return data[data['BAT_EVENT_FL'] == 'T'].reset_index(drop=True)


def encode_teams(data):
    data['HOME_TEAM_ID'] = data['GAME_ID'].apply(lambda x: x[:3])
    teams = np.unique(
        data[['AWAY_TEAM_ID', 'HOME_TEAM_ID']].values.reshape(-1))
    encoder = OrdinalEncoder().fit(teams.reshape(-1, 1))
    data['AWAY_TEAM_ID'] = encoder.transform(
        data['AWAY_TEAM_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    data['HOME_TEAM_ID'] = encoder.transform(
        data['HOME_TEAM_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    return data


def encode_batters(data):
    data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']] = \
        data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']]\
        .fillna('')
    batters = np.unique(
        data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']]
        .values.reshape(-1))
    encoder = OrdinalEncoder().fit(batters.reshape(-1, 1))
    data['BAT_ID'] = encoder.transform(
        data['BAT_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    data['BASE1_RUN_ID'] = encoder.transform(
        data['BASE1_RUN_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    data['BASE2_RUN_ID'] = encoder.transform(
        data['BASE2_RUN_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    data['BASE3_RUN_ID'] = encoder.transform(
        data['BASE3_RUN_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    return data


def encode_pitchers(data):
    encoder = OrdinalEncoder()
    data['PIT_ID'] = encoder.fit_transform(
        data['PIT_ID'].values.reshape(-1, 1)).reshape(-1).astype(int)
    return data


def add_bat_ids(data):
    away_bat_ids_arr = np.zeros((len(data), 9), dtype=np.uint16)
    home_bat_ids_arr = np.zeros((len(data), 9), dtype=np.uint16)

    for index, row in data.iterrows():
        if row['GAME_NEW_FL'] == 'T':  # Initialize
            game_new_index = index
            away_bat_ids, home_bat_ids = [], []
            have_away_bat_id, have_home_bat_id = False, False

        if row['BAT_HOME_ID'] == 0:  # Away team batting
            if len(away_bat_ids) == 9:
                away_bat_ids[row['BAT_LINEUP_ID'] - 1] = row['BAT_ID']
                if not have_away_bat_id:
                    have_away_bat_id = True

                    bat_lineup_id = 1
                    for j in range(game_new_index, index):
                        if data['BAT_HOME_ID'].iat[j] == 0:
                            bat_lineup_id = data['BAT_LINEUP_ID'].iat[j]
                        away_bat_ids_arr[j] = \
                            away_bat_ids[bat_lineup_id-1:] + \
                            away_bat_ids[:bat_lineup_id-1]
            else:
                away_bat_ids.append(row['BAT_ID'])
        else:                        # Home team batting
            if len(home_bat_ids) == 9:
                home_bat_ids[row['BAT_LINEUP_ID'] - 1] = row['BAT_ID']
                if not have_home_bat_id:
                    have_home_bat_id = True

                    bat_lineup_id = 1
                    for j in range(game_new_index, index):
                        if data['BAT_HOME_ID'].iat[j] == 1:
                            bat_lineup_id = data['BAT_LINEUP_ID'].iat[j]
                        home_bat_ids_arr[j] = \
                            home_bat_ids[bat_lineup_id-1:] + \
                            home_bat_ids[:bat_lineup_id-1]
            else:
                home_bat_ids.append(row['BAT_ID'])
        if len(away_bat_ids) == 9:
            away_bat_ids_arr[index] = away_bat_ids[row['BAT_LINEUP_ID']-1:] \
                + away_bat_ids[:row['BAT_LINEUP_ID']-1]
        if len(home_bat_ids) == 9:
            home_bat_ids_arr[index] = home_bat_ids[row['BAT_LINEUP_ID']-1:] \
                + home_bat_ids[:row['BAT_LINEUP_ID']-1]

    for i in range(9):
        data[f'AWAY_BAT{i+1}_ID'] = away_bat_ids_arr[:, i]
        data[f'HOME_BAT{i+1}_ID'] = home_bat_ids_arr[:, i]
    return data


def add_pit_ids(data):
    away_pit_ids = np.zeros(len(data), dtype=np.uint16)
    home_pit_ids = np.zeros(len(data), dtype=np.uint16)

    for index, row in data.iterrows():
        if row['GAME_NEW_FL'] == 'T':  # Initialize
            game_new_index = index
            away_pit_id, home_pit_id = 0, 0
            have_away_pit_id, have_home_pit_id = False, False

        if row['BAT_HOME_ID'] == 0:  # Home team pitching
            home_pit_id = row['PIT_ID']
            if not have_home_pit_id:
                have_home_pit_id = True
                home_pit_ids[game_new_index:index] = home_pit_id
        else:                        # Away team pitching
            away_pit_id = row['PIT_ID']
            if not have_away_pit_id:
                have_away_pit_id = True
                away_pit_ids[game_new_index:index] = away_pit_id
        if have_home_pit_id:
            home_pit_ids[index] = home_pit_id
        if have_away_pit_id:
            away_pit_ids[index] = away_pit_id

    data['AWAY_PIT_ID'] = away_pit_ids
    data['HOME_PIT_ID'] = home_pit_ids
    return data


def add_bat_lineup_ids(data):
    away_bat_lineup_ids, home_bat_lineup_ids = [], []

    for index, row in data.iterrows():
        if row['GAME_NEW_FL'] == 'T':  # Initialize
            away_bat_lineup_id, home_bat_lineup_id = 1, 1

        if row['BAT_HOME_ID'] == 0:  # Away team batting
            away_bat_lineup_id = row['BAT_LINEUP_ID']
        else:                        # Home team batting
            home_bat_lineup_id = row['BAT_LINEUP_ID']
        away_bat_lineup_ids.append(away_bat_lineup_id)
        home_bat_lineup_ids.append(home_bat_lineup_id)

    data['AWAY_BAT_LINEUP_ID'] = away_bat_lineup_ids
    data['HOME_BAT_LINEUP_ID'] = home_bat_lineup_ids
    return data


def add_end_score_cts(data):
    away_end_score_cts = np.zeros(len(data), dtype=np.uint8)
    home_end_score_cts = np.zeros(len(data), dtype=np.uint8)

    game_new_index = 0
    for index, row in data.iterrows():
        # Initialize
        if row['GAME_NEW_FL'] == 'T':
            game_new_index = index

        # AWAY_END_SCORE_CT, HOME_END_SCORE_CT
        if row['GAME_END_FL'] == 'T':
            event_runs_ct = (row[['BAT_DEST_ID', 'RUN1_DEST_ID',
                             'RUN2_DEST_ID', 'RUN3_DEST_ID']] >= 4).sum()
            if row['BAT_HOME_ID'] == 0:
                away_end_score_ct = row['AWAY_SCORE_CT'] + event_runs_ct
                home_end_score_ct = row['HOME_SCORE_CT']
            else:
                away_end_score_ct = row['AWAY_SCORE_CT']
                home_end_score_ct = row['HOME_SCORE_CT'] + event_runs_ct
            away_end_score_cts[game_new_index:index+1] = away_end_score_ct
            home_end_score_cts[game_new_index:index+1] = home_end_score_ct
            game_new_index = index + 1

    data['AWAY_END_SCORE_CT'] = away_end_score_cts
    data['HOME_END_SCORE_CT'] = home_end_score_cts
    return data


def add_end_inns(data):
    away_bat_inns = np.zeros(len(data), dtype=np.uint8)
    home_bat_inns = np.zeros(len(data), dtype=np.uint8)

    game_new_index = 0
    for index, row in data.iterrows():
        # Initialize
        if row['GAME_NEW_FL'] == 'T':
            game_new_index = index

        # AWAY_BAT_INN, HOME_BAT_INN
        if row['GAME_END_FL'] == 'T':
            if row['BAT_HOME_ID'] == 0:
                away_bat_inn = row['INN_CT']
                home_bat_inn = row['INN_CT'] - 1
            else:
                away_bat_inn = row['INN_CT']
                home_bat_inn = row['INN_CT']
            away_bat_inns[game_new_index:index+1] = away_bat_inn
            home_bat_inns[game_new_index:index+1] = home_bat_inn
            game_new_index = index + 1

    data['AWAY_BAT_INN'] = away_bat_inns
    data['HOME_BAT_INN'] = home_bat_inns

    return data


def preprocess(data):
    used_columns = [
        'GAME_ID',
        'GAME_NEW_FL',
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
        'HOME_END_SCORE_CT',
        'EVENT_CD',
        'AWAY_BAT_INN',
        'HOME_BAT_INN'
    ] + [f'AWAY_BAT{i}_ID' for i in range(1, 10)] \
      + [f'HOME_BAT{i}_ID' for i in range(1, 10)]

    # data = leave_only_bat_events(data)
    data = encode_teams(data)
    data = encode_batters(data)
    data = encode_pitchers(data)
    data = add_bat_ids(data)
    data = add_pit_ids(data)
    data = add_bat_lineup_ids(data)
    data = add_end_score_cts(data)
    data = add_end_inns(data)
    return data[used_columns]


def save_data_info(data):
    ''' Make and save data_info.json. '''

    data['HOME_TEAM_ID'] = data['GAME_ID'].apply(lambda x: x[:3])

    data_info = {
        'n_batters': len(np.unique(
            data[['BAT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID']]
            .fillna('').values.reshape(-1))),
        'n_pitchers': len(np.unique(data['PIT_ID'])),
        'n_teams': len(np.unique(
            data[['AWAY_TEAM_ID', 'HOME_TEAM_ID']].values.reshape(-1))),
        'events': list(map(int, np.unique(data['EVENT_CD'].values)))
    }

    with open('data_info.json', 'w') as f:
        json.dump(data_info, f)


def split_by_game(data):
    '''
    Return:
        [(t1, v1), ..., (tn, vn)], test
    '''

    non_test_game_ids, test_game_ids = \
        train_test_split(data['GAME_ID'].unique(), test_size=0.15)

    test_data = data[data['GAME_ID'].isin(test_game_ids)]

    # 5-fold cross validation
    non_test_data = []
    kf = KFold(n_splits=5)
    for train_indice, valid_indice in kf.split(non_test_game_ids):
        train_game_ids = [non_test_game_ids[idx] for idx in train_indice]
        valid_game_ids = [non_test_game_ids[idx] for idx in valid_indice]

        train_data = data[data['GAME_ID'].isin(train_game_ids)]
        valid_data = data[data['GAME_ID'].isin(valid_game_ids)]

        non_test_data.append((train_data, valid_data))

    return non_test_data, test_data


def drop_rows(data):
    return data[data['GAME_NEW_FL'] == 'T']


def split_data(data):
    non_test_data, test_data = split_by_game(data)
    # valid_data = drop_rows(valid_data)
    test_data = drop_rows(test_data)
    return non_test_data, test_data


def drop_cols(data):
    columns_to_drop = ['GAME_ID']
    return data.drop(columns_to_drop, 'columns')


def main():
    data = pd.read_csv('input/mlbplaybyplay2010s/all2010.csv',
                       low_memory=False)

    save_data_info(data)

    data = preprocess(data)

    non_test_data, test_data = split_data(data)

    for i, (train_data, valid_data) in enumerate(non_test_data):
        train_data = drop_cols(train_data)
        valid_data = drop_cols(valid_data)

        train_data.to_csv(
            f'input/mlbplaybyplay2010s_preprocessed/all2010_train_{i}.csv',
            index=False
        )
        valid_data.to_csv(
            f'input/mlbplaybyplay2010s_preprocessed/all2010_valid_{i}.csv',
            index=False
        )

    test_data = drop_cols(test_data)

    test_data.to_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_test.csv', index=False)


if __name__ == '__main__':
    main()
