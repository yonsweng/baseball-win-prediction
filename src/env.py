import torch


class Env():
    def __init__(self):
        return

    def get_state(self):
        return {
            'outs_ct': torch.tensor([[self.state['outs_ct']]], dtype=torch.float),
            'bat_id': torch.tensor([[self.state['bat_id']]], dtype=torch.long),
            'pit_id': torch.tensor([[self.state['pit_id']]], dtype=torch.long),
            'fld_team_id': torch.tensor([[self.state['fld_team_id']]], dtype=torch.long),
            'base1_run_id': torch.tensor([[self.state['base1_run_id']]], dtype=torch.long),
            'base2_run_id': torch.tensor([[self.state['base2_run_id']]], dtype=torch.long),
            'base3_run_id': torch.tensor([[self.state['base3_run_id']]], dtype=torch.long),
            'away_score_ct': torch.tensor([[self.state['away_score_ct']]], dtype=torch.float),
            'home_score_ct': torch.tensor([[self.state['home_score_ct']]], dtype=torch.float),
            'inn_ct': torch.tensor([[self.state['inn_ct']]], dtype=torch.float),
            'bat_home_id': torch.tensor([[self.state['bat_home_id']]], dtype=torch.float),
            'away_bat_lineup': torch.tensor([[self.state['away_bat_lineup']]], dtype=torch.float),
            'home_bat_lineup': torch.tensor([[self.state['home_bat_lineup']]], dtype=torch.float),
            'away_start_bat_ids': torch.tensor([self.away_bat_ids], dtype=torch.long),
            'home_start_bat_ids': torch.tensor([self.home_bat_ids], dtype=torch.long),
            'away_pit_id': torch.tensor([[self.away_pit_id]], dtype=torch.long),
            'home_pit_id': torch.tensor([[self.home_pit_id]], dtype=torch.long),
            'away_team_id': torch.tensor([[self.away_team_id]], dtype=torch.long),
            'home_team_id': torch.tensor([[self.home_team_id]], dtype=torch.long)
        }

    def reset(self, state, targets):
        self.away_bat_ids = state['away_start_bat_ids'][0].tolist()
        self.home_bat_ids = state['home_start_bat_ids'][0].tolist()
        self.away_pit_id = state['away_pit_id'][0].item()
        self.home_pit_id = state['home_pit_id'][0].item()
        self.away_team_id = state['away_team_id'][0].item()
        self.home_team_id = state['home_team_id'][0].item()
        self.result = targets['result']
        self.pred_len = 0
        self.state = {
            'inn_ct': int(state['inn_ct'][0].item()),
            'bat_home_id': int(state['bat_home_id'][0].item()),
            'outs_ct': int(state['outs_ct'][0].item()),
            'bat_id': state['bat_id'][0].item(),
            'base1_run_id': state['base1_run_id'][0].item(),
            'base2_run_id': state['base2_run_id'][0].item(),
            'base3_run_id': state['base3_run_id'][0].item(),
            'pit_id': state['pit_id'][0].item(),
            'fld_team_id': state['fld_team_id'][0].item(),
            'away_bat_lineup': int(state['away_bat_lineup'][0].item()),
            'home_bat_lineup': int(state['home_bat_lineup'][0].item()),
            'away_score_ct': int(state['away_score_ct'][0].item()),
            'home_score_ct': int(state['home_score_ct'][0].item())
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
            self.state['away_bat_lineup'] = self.state['away_bat_lineup'] % 9 + 1
        else:
            self.state['home_bat_lineup'] = self.state['home_bat_lineup'] % 9 + 1

        event_score_ct = (act_bat >= 4) + (act_run1 >= 4) + (act_run2 >= 4) + (act_run3 >= 4)
        if self.state['bat_home_id'] == 0:
            self.state['away_score_ct'] += event_score_ct
        else:
            self.state['home_score_ct'] += event_score_ct

        self.state['outs_ct'] += ((act_bat == 0)
            + int(self.state['base1_run_id'] != 0 and act_run1 == 0)
            + int(self.state['base2_run_id'] != 0 and act_run2 == 0)
            + int(self.state['base3_run_id'] != 0 and act_run3 == 0))

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
            result = 1 if self.state['away_score_ct'] < self.state['home_score_ct'] else 0
            reward = 1.0 if result == self.result else -1.0
        else:
            reward = 0.0

        if self.state['bat_home_id'] == 0:
            self.state['bat_id'] = self.away_bat_ids[self.state['away_bat_lineup'] - 1]
        else:
            self.state['bat_id'] = self.home_bat_ids[self.state['home_bat_lineup'] - 1]

        return self.get_state(), reward, done
