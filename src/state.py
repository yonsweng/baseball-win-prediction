''' GameState '''
class GameState():
    ''' GameState '''
    def __init__(self):
        self.inn_ct = 1
        self.bat_home_id = 0
        self.away_bat_lineup = 0
        self.home_bat_lineup = 0
        self.done = False

    def step(self, inn_end, away_score_ct, home_score_ct):
        '''
        Returns:
            inn_ct,
            bat_home_id,
            bat_lineup,
            done
        '''
        if inn_end:
            self.inn_end(away_score_ct, home_score_ct)

        # n(>=9)회말에 끝내기가 나와도 끝내야 한다.
        if self.inn_ct >= 9 and away_score_ct < home_score_ct:
            self.done = True

        if self.bat_home_id == 0:
            self.away_bat_lineup = self.away_bat_lineup % 9 + 1
            bat_lineup = self.away_bat_lineup
        else:
            self.home_bat_lineup = self.home_bat_lineup % 9 + 1
            bat_lineup = self.home_bat_lineup

        return self.inn_ct,\
               self.bat_home_id,\
               bat_lineup,\
               self.done

    def inn_end(self, away_score_ct, home_score_ct):
        '''
        Return:
            None
        '''
        if self.bat_home_id == 0:
            self.bat_home_id = 1
            if self.inn_ct >= 9 and away_score_ct < home_score_ct:
                self.done = True
        else:
            self.inn_ct += 1
            self.bat_home_id = 0
            if self.inn_ct > 9 and away_score_ct != home_score_ct:
                self.done = True
        if self.inn_ct > 12:
            self.done = True
