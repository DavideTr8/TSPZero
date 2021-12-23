import math
import numpy as np

from Game import Game
from NeuralNetwork import NeuralNetwork
import mcts_params


class MCTS:
    def __init__(self, game: Game, nn: NeuralNetwork, shortest_path: float):
        self.game = game
        self.nn = nn
        self.shortest_path = shortest_path
        self.Ps = {}
        self.Qsa = {}
        self.Ns = {}
        self.Nsa = {}

    def search(self, state=None):
        if state is None:
            state = self.game.state

        valid_moves = self.game.available_actions()
        if self.game.game_over():
            return self.game.score(self.shortest_path)

        if state not in self.Ps:
            self.Ps[state], v = self.nn(state)
            self.Ps[state] = [self.Ps[state][x] if x in valid_moves else 0 for x in range(self.game.n_nodes)]
            self.Ps[state] /= sum(self.Ps[state])

            self.Ns[state] = 0

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for action in range(self.game.n_nodes):
            if action in valid_moves:
                if (state, action) in self.Qsa:
                    u = self.Qsa[(state, action)] + mcts_params.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state]) / (
                            1 + self.Nsa[(state, action)])
                else:
                    u = mcts_params.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state] + mcts_params.EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = action

        action = best_act
        new_state = self.game.step(action)

        v = self.search(new_state)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + v) / (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = v
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return v

    def getActionProb(self, num_simulations, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(num_simulations):
            self.search()

        s = self.game.state
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.n_nodes)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
