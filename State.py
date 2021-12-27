import numpy as np


class State:
    def __init__(self, current_node: int, node_distances: np.array, visited_nodes: list[int]):
        self.current_node = current_node
        self.node_distances = node_distances
        self.visited_nodes = visited_nodes

    def __len__(self):
        return self.node_distances.shape[0] + 1
