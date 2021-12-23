import numpy as np


class Game:
    def __init__(self, n_nodes: int, distances: np.array, state: tuple = (0,)):
        """
        A VRP game where each state is labelled by an integer value ranging from 0 to n_nodes,
        the state is represented as the list of the already visited nodes and the distance of the tour is
        computed using the distances matrix.

        :param n_nodes: int, number of nodes in the VRP.
        :param distances: distances between each pair of nodes.
        """
        self.n_nodes: int = n_nodes
        self.distances: np.array = distances
        self.state: tuple[int] = tuple() if state is None else state

    def available_actions(self) -> list[int]:
        """Returns the available actions from the given state."""
        return [x for x in range(self.n_nodes) if x not in self.state]

    def game_over(self) -> bool:
        """Returns if the state of the game is a final state or not."""
        return len(self.state) == self.n_nodes

    def step(self, action: int):
        """Return the state obtained appending the selected node to visit."""
        new_state = (*self.state, action)
        return Game(self.n_nodes, self.distances, new_state)

    def score(self, opponent_distance: float) -> int:
        """
        Return 1 if the total travelled distance is better than the opponent one, else returns -1.
        :param opponent_distance: float, distance travelled by the opponent (can be the distance found using a
        heuristic).
        :return: 1 or -1, 1 for winning and -1 for losing.
        """
        if not self.game_over():
            raise Exception("The score of an unfinished game is trying to be computed.")

        distance = self.get_path_len()

        if distance <= opponent_distance:
            return 1
        return -1

    def get_path_len(self):
        distance = self.distances[self.state[-1], 0]
        for node_idx in range(self.n_nodes - 1):
            distance += self.distances[self.state[node_idx], self.state[node_idx + 1]]

        return distance
