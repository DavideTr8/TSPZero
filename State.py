import numpy as np
import torch
import torch.nn.functional as F

from utils import adjacency_matrix

class State:
    def __init__(self, current_node: int, distances: np.array, visited_nodes: list[int]):
        self.current_node = current_node
        self.distances = distances
        self.visited_nodes = visited_nodes

    def __len__(self):
        return self.distances.shape[0]

    def to_onehot(self):
        one_hot = F.one_hot(torch.tensor([self.current_node]), num_classes=len(self))
        return one_hot

    def mask_distances(self):
        A = adjacency_matrix(len(self), self.visited_nodes[:-1])
        D_prime = self.distances * A
        return D_prime



