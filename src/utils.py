def count_numbers(tmp):
    '''
    Returns:
        num_bats,
        num_pits,
        num_teams
    '''
    bats = list(tmp['BAT_ID']) \
         + list(tmp['BASE1_RUN_ID']) \
         + list(tmp['BASE2_RUN_ID']) \
         + list(tmp['BASE3_RUN_ID'])
    return len(set(bats)), \
           len(tmp['PIT_ID'].unique()), \
           len(tmp['AWAY_TEAM_ID'].unique())
