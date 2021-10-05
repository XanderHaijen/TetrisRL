from abc import abstractmethod
from typing import Callable
from tetris_environment.tetris_env import TetrisEnv


class StateValueModel:
    def __init__(self, env: TetrisEnv):
        self.env = env

    @abstractmethod
    def train(self, learning_rate: Callable[[int], float], nb_episodes: int = 1000, start_episode: int = 0) -> None:
        """
        Implemented by each model
        """
        pass

    @abstractmethod
    def predict(self, state):
        """
        Implemented by each model
        :return: the best action to take according to the model. Does not explore, only exploit.
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
    def load(filename: str, rendering: bool):
        pass

    @abstractmethod
    def __str__(self):
        pass
