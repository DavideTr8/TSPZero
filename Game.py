import numpy as np
from State import State

class Game:
    def __init__(self, distances: np.array):
        """
        The game is a TSP. A TSP can be described as a complete graph where each node is numbered with
        numbers 0, ..., n_nodes -1 and each edge (i, j) has as attribute the distance between nodes i and j.
        That is the Game is fully described by providing the distance matrix between nodes.

        :param distances: np.array, distance matrix.
        """
        self.distances = distances
        self.n_nodes = self.distances.shape[0]
        self.all_actions = [x for x in range(self.n_nodes)]

    def available_actions(self, state: State) -> list[int]:
        """
        Returns the list of available action at a given state.

        :param state: State, state of the game.
        :return: list, all available actions.
        """
        return [a for a in range(self.n_nodes) if a not in state.visited_nodes]

    def game_over(self, state: State) -> bool:
        """
        True if the state is a final state, i.e. the game is over, False otherwise.

        :param state: State, state of the game.
        :return: bool, True for game over, False otherwise.
        """
        return self.n_nodes == len(state.visited_nodes)

    def step(self, state: State, action: int) -> State:
        """
        Gives the new state achieved by performing the passed action from state.

        :param state: State, state of the game.
        :param action: int, action to be performed. Should be a feasible action.
        :return: State, the new state reached.
        """
        visited_nodes = (*state.visited_nodes, action)
        return State(action, self.distances, visited_nodes)

    def get_objective(self, state: State) -> float:
        """
        Give the lenght of the tour if the state is a final state, else raise an error.
        :param state: State, state of the game.
        :return: float, total length of the tour.
        """
        if not self.game_over(state):
            raise Exception("The objective of a partial solution is trying to be computed.")

        obj = self.distances[state.current_node, 0]
        for node_idx in range(self.n_nodes - 1):
            obj += self.distances[state.visited_nodes[node_idx], state.visited_nodes[node_idx + 1]]

        return obj

    def score(self, state: State, opponent_objective: float) -> int:
        """
        Return if the game is won or not by the player. 1 if the game is won, -1 otherwise.

        :param state: State, state of the game.
        :param opponent_objective: float, lenght of the tour found by the opponent.
        :return: int, 1 if the lenght of the tour of the player is less or equal to the opponent's, -1 otherwise.
        """
        player_objective = self.get_objective(state)
        if player_objective < opponent_objective:
            return 1
        return -1

