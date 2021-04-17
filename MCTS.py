# MIT License
# https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

import math
import numpy as np
from Env import Env
from utils import to_input

EPS = 1e-8


class MCTS:
    def __init__(self, device, nnet, action_space, args):
        self.env = Env(action_space)
        self.device = device
        self.nnet = nnet
        self.action_space = action_space
        self.args = args

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def get_policy(self, state) -> np.array:
        for _ in range(self.args.num_mcts_sims):
            self.search(state, self.args.mcts_max_depth)

        s = str(state)  # the string representation of state
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
                  for a in range(self.args.n_actions)]

        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return np.array(probs, dtype=np.float32)

    def search(self, state, remaining_depth):
        state = self.env.reset(state)
        s = str(state)

        if s not in self.Es:
            self.Es[s] = self.env.check_done(state)

        # terminal node
        if self.Es[s] is not None:  # if the game is done.
            return self.Es[s]

        # max depth
        if remaining_depth == 0:
            _, v = self.nnet.predict(to_input(state, self.device))
            return self.transform_value(state, v)

        # leaf node
        if s not in self.Ps:  # if s is not visited yet.
            self.Ps[s], v = self.nnet.predict(to_input(state, self.device))
            v = self.transform_value(state, v)
            valids = self.action_space.get_valid_moves(state)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            # return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.args.n_actions):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] \
                        * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] \
                        * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, _, _, _ = self.env.step(a)

        v = self.search(next_s, remaining_depth - 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) \
                               / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

    def transform_value(self, state, value):
        ''' (end_score - (future_score + curr_score)) ** 2 '''
        return - (state['AWAY_END_SCORE_CT'] -
                  (state['AWAY_SCORE_CT'] + value[0])) ** 2 \
               - (state['HOME_END_SCORE_CT'] -
                  (state['HOME_SCORE_CT'] + value[1])) ** 2
