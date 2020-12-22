import os
import torch


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


def get_latest_file_path(folder, prefix):
    files_Path = folder  # 파일들이 들어있는 폴더
    file_name_and_time_lst = []
    for f_name in os.listdir(f"{files_Path}"):
        if f_name.startswith(prefix):
            written_time = os.path.getctime(os.path.join(files_Path, f_name))
            file_name_and_time_lst.append((f_name, written_time))
    sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)
    recent_file = sorted_file_lst[0]
    recent_file_name = recent_file[0]
    return os.path.join(folder, recent_file_name)


def get_next_bats(bat_ids, bat_lineup):
    c = torch.cat([(bat_lineup - 1 + i) % 9 for i in range(1, 10)], 1)
    return bat_ids.gather(1, c)
