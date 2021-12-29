import numpy as np
import torch

class State:
    def __init__(self, current_node: int, node_distances: np.array, visited_nodes: list[int]):
        self.current_node = current_node
        self.node_distances = node_distances
        self.visited_nodes = visited_nodes

    def __len__(self):
        return self.node_distances.shape[0] + 1

    def to_tensor(self):
        as_list = [self.current_node] + self.node_distances.tolist()
        return torch.tensor(as_list)

    def get_identifier(self):
        return self.visited_nodes
