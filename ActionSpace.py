class ActionSpace:
    def __init__(self):
        self.dests_to_action = {}
        self.action_to_dests = {}

        self.action_cnt = 0

        for dest0 in range(5):
            for dest1 in range(5):
                for dest2 in range(5):
                    for dest3 in range(5):
                        dests = (dest0, dest1, dest2, dest3)
                        if self.is_valid(dests):
                            self.dests_to_action[dests] = self.action_cnt
                            self.action_to_dests[self.action_cnt] = dests
                            self.action_cnt += 1

        # print(self.action_to_dests)

    def is_valid(self, dests):
        # two or more 4s can exist.
        lowest = 4
        for i, dest in enumerate(dests[::-1]):
            # ignore when dest is 0.
            if dest == 0:
                continue

            # going back is not allowed.
            if dest < 3 - i:
                return False

            # if lowest is 4, then any dest is allowed.
            if lowest == 4:
                lowest = dest
            else:
                if dest >= lowest:
                    return False
                lowest = dest

        return True

    def __len__(self):
        return self.action_cnt

    def to_action(self, dests):
        return self.dests_to_action[dests]

    def to_dests(self, action):
        return self.action_to_dests[action]
