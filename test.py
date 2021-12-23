from trainer import Trainer
from Game import Game
from NeuralNetwork import NeuralNetwork
import numpy as np

np.random.seed(42)
n_nodes = 20
A = np.random.rand(n_nodes, n_nodes)
distances = (A.T * A) / 2

game = Game(n_nodes, distances)
nnet = NeuralNetwork(n_nodes)

coach = Trainer(game, nnet)
coach.learn()
