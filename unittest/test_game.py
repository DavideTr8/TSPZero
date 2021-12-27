import unittest

import numpy as np

from Game import Game
from State import State

np.random.seed(42)

n_nodes = 4
A = np.random.rand(n_nodes, n_nodes)

class TestAvailableActions(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = (A.T * A) / 2
        self.game = Game(self.distances)

    def test_available_actions(self):
        state = State(1, self.distances[:, 1], [0, 1])
        avail_expected = [2, 3]
        avail_returned = self.game.available_actions(state)
        self.assertEqual(avail_returned, avail_expected)

    def test_no_actions(self):
        state = State(1, self.distances[:, 1], [0, 3, 2, 1])
        avail_expected = []
        avail_returned = self.game.available_actions(state)
        self.assertEqual(avail_returned, avail_expected)

class TestGameOver(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = (A.T * A) / 2
        self.game = Game(self.distances)

        self.fixtures = (
            (State(1, self.distances[:, 1], [0, 1]), False),
            (State(1, self.distances[:, 1], [0, 3, 2, 1]), True),
        )

    def test_fixtures(self):
        for state, expected in self.fixtures:
            if expected:
                self.assertTrue(self.game.game_over(state))
            else:
                self.assertFalse(self.game.game_over(state))


class TestStep(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = (A.T * A) / 2
        self.game = Game(self.distances)

        self.fixtures = (
            (State(1, self.distances[:, 1], [0, 1]), 2, State(2, self.distances[:, 2], [0, 1, 2])),
            (State(1, self.distances[:, 1], [0, 1]), 3, State(3, self.distances[:, 3], [0, 1, 3])),
        )

    def test_fixtures(self):
        for state, action, expected in self.fixtures:
            new_state = self.game.step(state, action)
            self.assertEqual(new_state.current_node, expected.current_node)
            self.assertTrue(all(new_state.node_distances == expected.node_distances))
            self.assertEqual(new_state.visited_nodes, expected.visited_nodes)


class TestGetObjective(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7]
            ]
        )
        self.game = Game(self.distances)
        self.state = State(3, self.distances[:, 3], [0, 1, 2, 3])

    def test_objective(self):
        obj_returend = self.game.get_objective(self.state)
        obj_expected = 16
        self.assertEqual(obj_expected, obj_returend)


class TestScore(unittest.TestCase):
    def setUp(self) -> None:
        self.distances = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7]
            ]
        )
        self.game = Game(self.distances)
        self.state = State(3, self.distances[:, 3], [0, 1, 2, 3])
        self.fixtures = (
            (15, -1),
            (16, 1),
            (17, 1),
        )

    def test_fixtures(self):
        for opponent_obj, score_expected in self.fixtures:
            score_returend = self.game.score(self.state, opponent_obj)
            self.assertEqual(score_returend, score_expected)


if __name__ == "__main__":
    unittest.main()
