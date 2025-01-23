class TeamGame():
    def __init__(self, game_id, team_id):
        self.game_id = game_id
        self.team_id = team_id
        
        self.fgm = 0
        self.fga = 0
        self.fg3m = 0
        self.fg3a = 0
        self.ftm = 0
        self.fta = 0
        self.dreb = 0
        self.oreb = 0
        self.tov = 0
    
    def efg(self):
        return (self.fgm + 0.5 * self.fg3m) / self.fga
    
    def poss(self):
        return self.fga + (0.44 * self.fta) + self.tov - self.oreb
    
    def tov_pct(self):
        return self.tov / (self.fga + (0.44 * self.fta) + self.tov)
    
    def fg3_pct(self):
        return self.fg3m / self.fg3a
    
    def ftr(self):
        return self.fta / self.fga