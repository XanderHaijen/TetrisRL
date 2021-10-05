import random
from typing import Callable, Tuple
import pickle
from tetris_environment.tetris_env import TetrisEnv
from Models.AfterstateModel import AfterstateModel


class SarsaZeroAfterStates(AfterstateModel):
    """
    A Sarsa model working with an afterstate value function V(S')
    """

    def __init__(self, env: TetrisEnv, alpha: float = 1, gamma: float = 1, value_function: dict = None):
        """

        :param alpha: step-size-parameter in the update rule
        :param gamma: parameter in the update rule
        """
        super().__init__(env)

        # the value function is represented by a dict of states S'.
        # All values are initialized to 0.
        if value_function is None:
            value_function = {}

        self.value_function = value_function

        self.alpha = alpha
        self.gamma = gamma

    def train(self, learning_rate: Callable[[int], float], nb_episodes: int = 1000, start_episode: int = 0) -> None:
        """
        Updates the value function according to the Sarsa(0) model (Sutton & Barto, page 155)
        :param learning_rate: = epsilon. A function of the number of episodes which goes towards zero at infinity
        :param nb_episodes: the duration of ´´the training session´´
        :param start_episode: zero in the beginning, greater than zero when training an already (partially)
        trained agent
        :return:
        """
        for episode in range(1, nb_episodes + 1):
            state = self.env.reset()
            afterstate, actions = self._epsilon_greedy_actions(learning_rate, episode + start_episode)

            done = False
            while not done:
                reward = 0
                for action in actions:
                    # take action a, observe R and s' until the piece has reached the bottom
                    state, reward, done, obs = self.env.step(action)
                    self.env.render()

                afterstate, actions = self._epsilon_greedy_actions(learning_rate, episode + start_episode)

                # update value function at V(s)
                value_at_next_state = self.value_function.get(afterstate, 0)
                old_value = self.value_function.get(state, 0)
                new_value = old_value + self.alpha * (reward + self.gamma * value_at_next_state - old_value)
                self.value_function.update({state: new_value})

    def _epsilon_greedy_actions(self, learning_rate: Callable[[int], float], nb_episodes: int) -> \
            Tuple[tuple, list]:
        """
        Returns either a random set of actions leading to a random next board state, or a sequence of actions
        leading to the most favourable afterstate.
        :param learning_rate:
        :param nb_episodes:
        :return: a tuple of the form (afterstate, actions)
        """
        if random.random() < learning_rate(nb_episodes):
            return self._pick_random_actions()
        else:  # take greedy action
            return self.predict()

    def _nb_actions(self) -> int:
        return len(self.env.game_state.get_action_set())

    def _pick_random_actions(self) -> Tuple[tuple, list]:
        possible_placements = self.env.all_possible_placements()
        if len(possible_placements) > 0:
            placement = random.choice(possible_placements)  # consists of afterstate and action
        else:
            placement = (self.env.get_encoded_state(), [0])  # no piece, so no action
        return placement

    def predict(self) -> Tuple[tuple, list]:
        possible_placements = self.env.all_possible_placements()
        # possible_placements of form (state, action)
        if len(possible_placements) > 0:
            best_placement = max(possible_placements, key=lambda pl: self.value_function.get(pl[0], 0))
        else:
            best_placement = (self.env.get_encoded_state(), [0])  # no piece, so no action
        return best_placement

    @staticmethod
    def _load_file(filename: str):
        with open(filename, 'rb') as f:
            alpha, gamma, value_function, size = pickle.load(f)
        return alpha, gamma, value_function, size

    @staticmethod
    def load(filename: str, rendering: bool = False):
        alpha, gamma, value_function, size = SarsaZeroAfterStates._load_file(filename)
        env = TetrisEnv(type=size, render=rendering)
        return SarsaZeroAfterStates(alpha=alpha, gamma=gamma, value_function=value_function, env=env)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump((self.alpha, self.gamma, self.value_function, self.env.type), f)

    def __str__(self):
        return f"{self.env.type} Sarsa Zero afterstate model (alpha={self.alpha}, gamma={self.gamma})"
