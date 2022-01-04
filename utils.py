import numpy as np


def adjacency_matrix(dimension, visited_nodes):
    A = np.ones((dimension, dimension))
    A -= np.eye(dimension)
    A[visited_nodes, :] = A[:, visited_nodes] = 0

    return A