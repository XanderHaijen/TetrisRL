import random
from typing import Callable
from Models.StateActionModel import StateValueModel
import pickle

from tetris_environment.tetris_env import TetrisEnv


class SarsaZeroForTetris(StateValueModel):
    """
    A Sarsa model working with a state-action value function Q(s,a)
    """

    def __init__(self, env: TetrisEnv, alpha=1, gamma=1, value_function: dict = None):
        """

        :param alpha: step-size-parameter in the update rule
        :param gamma: parameter in the update rule
        """
        super().__init__(env)

        # the value function is represented by a dict of dicts. State-action pairs are stored as
        # {state: {action: value}}. Non-visited state-action pairs are not stored and their
        # values are considered zero in the beginning.
        # The size of this dict is thus num_states. Each nested dict has max length _nb_actions for each state
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

        for episode in range(nb_episodes):  # for each episode
            state = self.env.reset()
            piece = self.env.get_falling_piece()

            ext_state = (state, piece)
            action = self._epsilon_greedy_action(learning_rate, episode + start_episode, ext_state)
            done = False

            while not done:
                old_ext_state = ext_state  # save old state s
                old_action = action  # save old action a
                state, reward, done, obs = self.env.step(action)  # new state and action s', a'
                piece = self.env.get_falling_piece()
                if piece is not None:
                    ext_state = (state, piece)
                    action = self._epsilon_greedy_action(learning_rate, episode + start_episode, ext_state)

                    # update value function at Q(s,a)
                    value_at_next_state = self.value_function.get(ext_state, {}).get(action, 0)
                    old_value = self.value_function.get(old_ext_state, {}).get(old_action, 0)
                    new_value = old_value + self.alpha * (reward + self.gamma * value_at_next_state - old_value)
                    if new_value != 0:
                        if old_ext_state not in self.value_function.keys():
                            self.value_function.update({old_ext_state: {}})
                        self.value_function[old_ext_state].update({old_action: new_value})
                        print(self.value_function)
                else:  # if piece is None, there is no falling piece. Make no move
                    action = self.env.no_move

    def _epsilon_greedy_action(self, learning_rate: Callable[[int], float], nb_episodes, ext_state):
        """
        :param ext_state: the state for which to choose the epsilon greedy action
        :param nb_episodes: how far into learning is the agent
        :param learning_rate: a function of the number of episodes which goes towards zero at infinity
        :return: the action according to the epsilon greedy policy
        """
        epsilon = learning_rate(nb_episodes)
        if random.random() <= epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            action = self.predict(ext_state)
            return action

    def _nb_actions(self) -> int:
        return len(self.env.game_state.get_action_set())

    def predict(self, ext_state):

        values_for_state = self.value_function.get(ext_state, {})
        a_star = self._argmax_dict(values_for_state)  # A_star is the optimal action in state A
        return a_star

    def _argmax_dict(self, dc: dict):
        """
        Returns the / a key associated with the / a maximum value.
        :param dc:
        :return:
        """
        if len(dc.keys()) > 0:
            return max(dc.keys(), key=lambda key: dc.get(key, 0))
        else:  # in this case all values are zero, so argmax is the same as a random sample
            return self.env.action_space.sample()

    @staticmethod
    def _load_file(filename: str):
        with open(filename, 'rb') as f:
            alpha, gamma, type, value_function = pickle.load(f)
        return alpha, type, gamma, value_function

    @staticmethod
    def load(filename: str, rendering: bool = False):
        alpha, gamma, type, value_function = SarsaZeroForTetris._load_file(filename)
        return SarsaZeroForTetris(TetrisEnv(type, rendering), alpha, gamma, value_function)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump((self.alpha, self.gamma, self.env.type, self.value_function), f)

    def __str__(self):
        return f"{self.env.type} State-action Sarsa Zero model (alpha={self.alpha}, gamma={self.gamma})"
