from abc import abstractmethod
from typing import Callable

from tetris_environment.tetris_env import TetrisEnv


class Model:
    def __init__(self):
        self.env = TetrisEnv()

    @abstractmethod
    def train(self, learning_rate: Callable[[int], float], nb_episodes: int = 1000, start_episode: int = 0) -> None:
        """
        Implemented by each algorithm
        """
        pass

    @abstractmethod
    def predict(self, state):
        """
        Implemented by each algorithm
        :return: the best action to take according to the algorithm. Does not explore, only exploit.
        """
        pass

    @abstractmethod
    def _epsilon_greedy_action(self, learning_rate: Callable[[int], float], nb_episodes, state):
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str):
        pass
