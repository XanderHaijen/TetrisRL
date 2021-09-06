from abc import abstractmethod
from typing import Any
import pickle
from gym import Env


class Algorithm:

    @abstractmethod
    def train(self, learning_rate, nb_episodes=1000, start_episode=0):
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
    def save(self, filename: str):
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str):
        pass
