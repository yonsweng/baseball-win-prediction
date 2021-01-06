import torch


class Env():
    def __init__(self, features, targets):
        self.away_bat_ids = features['away_start_bat_ids'][0].tolist()
        self.home_bat_ids = features['home_start_bat_ids'][0].tolist()
        self.away_pit_id = features['away_start_pit_id'][0][0].item()
        self.home_pit_id = features['home_start_pit_id'][0][0].item()
        self.away_team_id = features['away_team_id'][0][0].item()
        self.home_team_id = features['home_team_id'][0][0].item()
        self.away_end_score = int(targets['value_away'][0][0].item())
        self.home_end_score = int(targets['value_home'][0][0].item())
        self.reset()

    def get_state(self):
        return {
            'away_score_ct': torch.Tensor([[self.state['away_score_ct']]]),
            'home_score_ct': torch.Tensor([[self.state['home_score_ct']]]),
            'inn_ct': torch.Tensor([[self.state['inn_ct']]]),
            'bat_home_id': torch.Tensor([[self.state['bat_home_id']]]),
            'outs_ct': torch.Tensor([[self.state['outs_ct']]]),
            'bat_id': torch.tensor([[self.state['bat_id']]], dtype=torch.long),
            'start_pit_id': torch.tensor([[self.state['pit_id']]], dtype=torch.long),
            'fld_team_id': torch.tensor([[self.state['fld_team_id']]], dtype=torch.long),
            'base1_run_id': torch.tensor([[self.state['base1_run_id']]], dtype=torch.long),
            'base2_run_id': torch.tensor([[self.state['base2_run_id']]], dtype=torch.long),
            'base3_run_id': torch.tensor([[self.state['base3_run_id']]], dtype=torch.long)
        }

    def reset(self):
        self.pred_len = 0
        self.away_bat_lineup = 1
        self.home_bat_lineup = 1
        self.state = {
            'inn_ct': 1,
            'bat_home_id': 0,
            'outs_ct': 0,
            'bat_id': self.away_bat_ids[0],
            'base1_run_id': 0,
            'base2_run_id': 0,
            'base3_run_id': 0,
            'pit_id': self.home_pit_id,
            'fld_team_id': self.home_team_id,
            'away_score_ct': 0,
            'home_score_ct': 0
        }
        return self.get_state()

    def switch(self):
        self.state['base1_run_id'] = self.state['base2_run_id'] = self.state['base3_run_id'] = 0
        self.state['inn_ct'] += self.state['bat_home_id']
        self.state['bat_home_id'] = 1 - self.state['bat_home_id']
        self.state['outs_ct'] = 0
        if self.state['bat_home_id'] == 0:
            self.state['pit_id'] = self.home_pit_id
            self.state['fld_team_id'] = self.home_team_id
        else:
            self.state['pit_id'] = self.away_pit_id
            self.state['fld_team_id'] = self.away_team_id

    def move(self, act_bat, act_run1, act_run2, act_run3):
        ''' 주자 옮기기 '''
        runners = [0, 0, 0, 0]
        runners[act_bat if act_bat < 4 else 0] = self.state['bat_id']
        runners[act_run1 if act_run1 < 4 else 0] = self.state['base1_run_id']
        runners[act_run2 if act_run2 < 4 else 0] = self.state['base2_run_id']
        runners[act_run3 if act_run3 < 4 else 0] = self.state['base3_run_id']
        self.state['base1_run_id'] = runners[1]
        self.state['base2_run_id'] = runners[2]
        self.state['base3_run_id'] = runners[3]

    def step(self, act_bat, act_run1, act_run2, act_run3):
        '''
        Returns:
            state,
            reward,
            done
        '''
        self.pred_len += 1

        if self.state['bat_home_id'] == 0:
            self.away_bat_lineup = self.away_bat_lineup % 9 + 1
        else:
            self.home_bat_lineup = self.home_bat_lineup % 9 + 1

        event_score_ct = (act_bat >= 4) + (act_run1 >= 4) + (act_run2 >= 4) + (act_run3 >= 4)
        if self.state['bat_home_id'] == 0:
            self.state['away_score_ct'] += event_score_ct
        else:
            self.state['home_score_ct'] += event_score_ct

        self.state['outs_ct'] += ((act_bat == 0)
            + (self.state['base1_run_id'] != 0 and act_run1 == 0)
            + (self.state['base2_run_id'] != 0 and act_run2 == 0)
            + (self.state['base3_run_id'] != 0 and act_run3 == 0))

        self.move(act_bat, act_run1, act_run2, act_run3)

        if self.state['outs_ct'] >= 3:
            self.switch()

        if (self.state['inn_ct'] >= 9 and self.state['bat_home_id'] == 1
            and self.state['away_score_ct'] < self.state['home_score_ct']) \
        or (self.state['inn_ct'] > 9 and self.state['bat_home_id'] == 0 and self.state['outs_ct'] == 0
            and self.state['away_score_ct'] != self.state['home_score_ct']) \
        or self.state['inn_ct'] > 12:
            done = True
        else:
            done = False

        if done:
            real_result = self.away_end_score < self.home_end_score
            pred_result = self.state['away_score_ct'] < self.state['home_score_ct']
            reward = 1 if real_result == pred_result else -1
        else:
            reward = 0

        if self.state['bat_home_id'] == 0:
            self.state['bat_id'] = self.away_bat_ids[self.away_bat_lineup - 1]
        else:
            self.state['bat_id'] = self.home_bat_ids[self.home_bat_lineup - 1]

        return self.get_state(), reward, done, None
