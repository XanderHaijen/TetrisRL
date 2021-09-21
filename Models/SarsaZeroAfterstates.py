import random
from typing import Callable
from Models.Model import Model
import pickle


class SarsaZeroAfterStates(Model):
    """
    A Sarsa model working with an afterstate value function V(S')
    """

    def __init__(self, alpha: float = 1, gamma: float = 1, value_function: dict = None):
        """

        :param alpha: step-size-parameter in the update rule
        :param gamma: parameter in the update rule
        """
        super().__init__()

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
            actions = self._epsilon_greedy_action(learning_rate, episode + start_episode, board=self.env.game_state)

            done = False
            while not done:
                old_state = state
                reward = 0
                for action in actions:
                    # take action a, observe R and s' until the piece has reached the bottom
                    state, reward, done, obs = self.env.step(action)

                actions = self._epsilon_greedy_action(learning_rate, episode + start_episode,
                                                      board=self.env.game_state)

                # update value function at V(s)
                value_at_next_state = self.value_function.get(state, 0)
                old_value = self.value_function.get(old_state, 0)
                new_value = old_value + self.alpha * (reward + self.gamma + value_at_next_state - old_value)
                self.value_function.update({state: new_value})

    def _epsilon_greedy_action(self, learning_rate: Callable[[int], float], nb_episodes: int, board):
        """
        Returns either a random set of actions leading to a random next board state, or a sequence of actions
        leading to the most favourable afterstate.
        :param learning_rate:
        :param nb_episodes:
        :param board:
        :return:
        """
        if random.random() < learning_rate(nb_episodes):
            return self._pick_random_action()
        else:  # take greedy action
            return self.predict(board)

    def _nb_actions(self) -> int:
        return len(self.env.game_state.get_action_set())

    def _pick_random_action(self):
        possible_placements = self.env.all_possible_placements()
        if len(possible_placements) > 0:
            afterstate, action = random.choice(possible_placements)
        else:
            action = (0,)  # no piece, no action
        return action

    def predict(self, board):
        possible_placements = self.env.all_possible_placements()
        # possible_placements of form (state, action)
        if len(possible_placements) > 0:
            best_placement = max(possible_placements, key=lambda pl: self.value_function.get(pl[0], 0))
            a_star = best_placement[1]
        else:
            a_star = (0,)  # no piece, so no action
        return a_star

    @staticmethod
    def _load_file(filename: str):
        with open(filename, 'rb') as f:
            alpha, gamma, value_function = pickle.load(f)
        return alpha, gamma, value_function

    @staticmethod
    def load(filename: str):
        alpha, gamma, value_function = SarsaZeroAfterStates._load_file(filename)
        return SarsaZeroAfterStates(alpha=alpha, gamma=gamma, value_function=value_function)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump((self.alpha, self.gamma, self.value_function), f)
