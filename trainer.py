import logging
from collections import deque
from random import shuffle

import numpy as np

from MCTS import MCTS
from Game import Game

log = logging.getLogger(__name__)


class Trainer:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, shortest_path=float("inf")):
        self.initial_game = game
        self.game = game
        self.nnet = nnet
        self.shortest_path = shortest_path
        self.mcts = MCTS(self.game, self.nnet, shortest_path)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.best_path = None

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < 30)
            self.mcts.game = self.game
            pi = self.mcts.getActionProb(50, temp=temp)
            trainExamples.append([np.array([*self.game.state, *[-1.0]*(self.game.n_nodes - len(self.game.state))]), pi, None])
            action = np.random.choice(len(pi), p=pi)
            self.game = self.game.step(action)

            if self.game.game_over():
                r = self.game.score(self.shortest_path)
                self.game = Game(self.game.n_nodes, self.game.distances)
                return [(x[0], x[1], r) for x in trainExamples]

    def play_game(self, strategy):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        it = 0
        while not self.game.game_over():
            it += 1
            print("Turn ", str(it))
            action = strategy(5)

            valids = self.game.available_actions()

            if action not in valids:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')

            self.game = self.game.step(action)
            self.mcts = MCTS(self.game, self.nnet, self.shortest_path)

        return self.game.get_path_len(), self.game.state

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        num_iter = 10
        num_episodes = 10
        for i in range(1, num_iter + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            self.game = Game(self.initial_game.n_nodes, self.initial_game.distances)
            iterationTrainExamples = deque([], maxlen=1000)

            for _ in range(num_episodes):
                self.mcts = MCTS(self.game, self.nnet, self.shortest_path)  # reset search tree
                iterationTrainExamples += self.execute_episode()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            # if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            #     log.warning(
            #         f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            #     self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.train(trainExamples)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            strategy = lambda x: np.argmax(self.mcts.getActionProb(x, temp=0))
            path_len, path = self.play_game(strategy)

            if path_len <= self.shortest_path:
                self.shortest_path = path_len
                self.best_path = path

            print(self.best_path, self.shortest_path)
