import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from Game import Game
from State import State

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, nnet, mcts, args):
        # self.game_instances = game_instances
        self.mcts = mcts
        self.nnet = nnet
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def execute_episode(self, game_instance):
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
        game = Game(game_instance.distances)
        state = State(0, game_instance.distances, (0, ))

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.get_policy(state, temp)

            trainExamples.append([state, pi, None])

            action = np.random.choice(len(pi), p=pi)
            state = game.step(state, action)

            if game.game_over(state):
                score = game.score(state, opponent_objective=game_instance.best_objective)
                return [(x[0], x[1], score) for x in trainExamples]

    def play_game(self, strategy, game_instance):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        it = 0
        state = State(0, game_instance.distances, (0, ))
        game = Game(distances=game_instance.distances)
        while not game.game_over(state):
            it += 1
            print("Turn ", str(it))
            action = strategy(state)

            valids = game.available_actions(state)

            if action not in valids:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')

            state = game.step(state, action)
            self.mcts = MCTS(game, self.nnet, game_instance.best_objective)

        return game.get_objective(state), state.visited_nodes

    def learn(self, game_instance):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            game = Game(game_instance.distances)
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(game, self.nnet, game_instance.best_objective)  # reset search tree
                iterationTrainExamples += self.execute_episode(game_instance)

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples, self.args.num_epoch)
            self.nmcts = MCTS(game, self.nnet, game_instance.best_objective)

            # test the new nn
            strategy = lambda x: np.argmax(self.nmcts.get_policy(x))
            path_len, path = self.play_game(strategy, game_instance)

            print(path_len, path)
            if path_len < game_instance.best_objective:
                game_instance.best_objective = path_len
                game_instance.best_path = path
            # log.info('ACCEPTING NEW MODEL')
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True