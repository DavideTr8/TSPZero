from collections import defaultdict

import numpy as np

from State import State
from Game import Game
from NeuralNetwork import NN

CPUCT = 1
NUM_SIMULATIONS = 100
TEMPERATURE = 1


class MCTS:
    def __init__(self, game: Game, nn: NN, opponent_objective: float):
        """
        MCTS class primarly has 2 methods. The search method that, starting from the input state, runs a simulation
        of the game choosing the actions by their Upper Confidence Bound value and updates the Ps, Qsa and Nsa values.

        A second method, get policy, is used to find an approximation for the optimal policy. This approximated optimal
        policy is used, during training time, as a ground truth to train the Neural Network. This approximated policy
        is computed starting from the Nsa values collected through many iterations of the search method.

        Ps is the prior distriution, given by a trained Neural Network.
        Qsa is the Q-value of the state-action pair, approximated using the value of the games played in many iterations
        of the search method.
        Nsa is the number of time the pair state-action in played. A high value in Nsa means that state-action pair
        was considered promising many times, a low value means it wasn't.

        :param game: Game, game that needs to be played.
        :param nn: NeuralNetwork, neural net that gives the prior distribution.
        :param opponent_objective: float, value to beat to decide if a game is won or not.
        """
        self.game = game
        self.nn = nn
        self.opponent_objective = opponent_objective
        self.Ps = {}
        self.Qsa = defaultdict(float)
        self.Nsa = defaultdict(int)

    def is_visited(self, state: State) -> bool:
        """Helper method to check if a state was already visited through search iterations."""
        return state.visited_nodes in self.Ps

    def search(self, state: State) -> float:
        """
        The search method collects values for Ps, Nsa and Qsa by playing a game until termination starting from state.

        This is a recursive method. Return the value of the game if the state reached is a terminal state for the
        game or if the state is unvisited (leaf node).
        Else, decide an action to perform, update the state by performing such action and then apply the search
        method to the new state. Then update the Nsa and Qsa value for the child.

        The action is chosen by computing the Upper Confidence Bound of each action and selecting the one with highest
        value. For computing the UCB, a probability distribution of the actions is needed. This probability distribution
        Ps is predicted by a Neural Network nn.

        Since only for the final state-action an exact value of the game is available, the value of each state
        is also approximated using the nn. This approximated value is used to compute an approximation of the Q-value
        Qsa. This Qsa is also updated at each iteration when a new approximation from a child node is available.

        :param state: State, state from which the tree in explored.
        :return: float, value of the game.
        """
        s = state.visited_nodes
        if self.game.game_over(state):
            return self.game.score(state, self.opponent_objective)

        if not self.is_visited(state):
            self.Ps[s], v = self.nn.predict(state)
            return v

        valid_moves = self.game.available_actions(state)

        best_u = -float("inf")
        best_a = None
        for a in valid_moves:
            u = self.Qsa[s, a] + (CPUCT * self.Ps[s][a] * np.sqrt(sum([self.Nsa[s, b] for b in valid_moves])) /
                                  (self.Nsa[s, a] + 1))
            if u >= best_u:
                best_u = u
                best_a = a

        new_state = self.game.step(state, best_a)
        v = self.search(new_state)
        self.Qsa[(s, best_a)] = (
                (self.Nsa[(s, best_a)] * self.Qsa[(s, best_a)] + v) /
                (self.Nsa[(s, best_a)] + 1)
        )
        self.Nsa[(s, best_a)] += 1

        return v

    def get_policy(self, state: State) -> list[float]:
        for _ in range(NUM_SIMULATIONS):
            self.search(state)

        s = state.visited_nodes

        if TEMPERATURE == 0:
            policy = [0] * self.game.n_nodes
            argmax = np.argmax([self.Nsa[s, a] for a in self.game.all_actions])
            policy[argmax] = 1
        else:
            policy = [self.Nsa[s, a] ** (1 / TEMPERATURE) for a in self.game.all_actions]
            sum_policy = sum(policy)
            policy /= sum_policy

        return policy
