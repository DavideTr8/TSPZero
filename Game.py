import numpy as np

class Game:
    def __init__(self, n_nodes: int, distances: np.array):
        """
        A VRP game where each state is labelled by an integer value ranging from 0 to n_nodes,
        the state is represented as the list of the already visited nodes and the distance of the tour is
        computed using the distances matrix.

        :param n_nodes: int, number of nodes in the VRP.
        :param distances: distances between each pair of nodes.
        """
        self.n_nodes: int = n_nodes
        self.distances: np.array = distances
        self.state: list[int] = []

    def available_actions(self) -> list[int]:
        """Returns the available actions from the given state."""
        return [x for x in range(self.n_nodes) if x not in self.state]

    def game_over(self) -> bool:
        """Returns if the state of the game is a final state or not."""
        return len(self.state) == self.n_nodes

    def step(self, action: int):
        """Update the state appending the selected node to visit."""
        self.state.append(action)

