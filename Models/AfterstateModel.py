from abc import abstractmethod
from typing import Callable, Tuple
from tetris_environment.tetris_env import TetrisEnv


class AfterstateModel:
    def __init__(self, env: TetrisEnv):
        self.env = env

    @abstractmethod
    def train(self, learning_rate: Callable[[int], float], nb_episodes: int = 1000, start_episode: int = 0) -> None:
        """
        Implemented by each model
        """
        pass

    @abstractmethod
    def predict(self) -> Tuple[tuple, list]:
        """
        Implemented by each model
        :return: the best action to take according to the model. Does not explore, only exploit.
        """
        pass

    @abstractmethod
    def _epsilon_greedy_actions(self, learning_rate: Callable[[int], float], nb_episodes: int) -> Tuple[tuple, list]:
        pass

    @abstractmethod
    def _pick_random_actions(self) -> Tuple[tuple, list]:
        pass

    @abstractmethod
    def save(self, filename: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _load_file(filename: str):
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str, rendering: bool = False):
        pass
