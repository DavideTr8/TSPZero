from Game import Game
from NeuralNetwork import NeuralNetwork

class MCTS:
    def __init__(self, game: Game, nn: NeuralNetwork):
        self.game = game
        self.nn = nn
        self.Ps = {}
        self.Qsa = {}
        self.Ns = {}
        self.Nsa = {}

    def search(self, state=None):
        if state is None:
            state = self.game.state

        valid_moves = self.game.available_actions()
        if self.game.game_over():
            return self.game.score()

        if state not in self.Ps:
            self.Ps[state], v = self.nn.predict(state)
            self.Ps[state] = [self.Ps[state][x] if x in valid_moves else 0 for x in range(self.game.n_nodes)]
            self.Ps[state] /= sum(self.Ps[state])

            self.Ns[state] = 0

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for action in range(self.game.n_nodes):
            if action in valid_moves:
                if (state, action) in self.Qsa:
                    u = self.Qsa[(state, action)] + self.args.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state]) / (
                            1 + self.Nsa[(state, action)])
                else:
                    u = self.args.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = action

        a = best_act
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