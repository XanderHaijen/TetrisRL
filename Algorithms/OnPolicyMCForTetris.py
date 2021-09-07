import pickle
import random
from typing import Callable

from Algorithms.Algorithm import Algorithm
from tetris_environment.tetris_env import TetrisEnv


class OnPolicyMCForTetris(Algorithm):
    def __init__(self, value_function: dict = None, cumul: dict = None, first_visit: bool = True):
        """
        Initializes a trainable Monte Carlo model.
        The model uses on-policy first-visit MC control for epsilon-soft policies. It does not
        require the assumption of exploring starts.
        :param value_function: a dict of dicts for all state-action pairs. If no value function is provided,
        the values are initialized as zero for all state-action pairs
        :param cumul: a dict containing the number of times a state-action pair has been visited

        """
        # the value function and cumul are represented by a dict of dicts. State-action pairs are stored as
        # {state: {action: value}}. Non-visited state-action pairs are not stored and their
        # values are considered zero in the beginning.
        # The size of this dict is thus num_states. Each nested dict has max length _nb_actions for each state
        if value_function is None:
            value_function = {}
        if cumul is None:
            cumul = {}

        self.value_function = value_function
        self.env = TetrisEnv()
        self.C = cumul
        self.first_visit = first_visit

    def train(self, learning_rate: Callable[[int], float], nb_episodes=1000,
              start_episode=0, gamma: float = 1) -> None:
        """
        Trains the MC model using on-policy MC control for epsilon soft policies (Sutton & Barto, sec. 5.4)
        :param gamma: Importance sampling factor. 0 < gamma < 1
        :ivar: 0 <= gamma <= 1
        :param first_visit: specifies whether the algorithm is first-visit or every-visit MC
        :param learning_rate: value of epsilon. Must become zero only in the limit, otherwise convergence
        is not guaranteed
        :param nb_episodes: number of episodes to train the agent
        :param start_episode: if working with an already (partially) trained agent, provide the number of already
        trained episodes
        :return: None
        """

        returns = {}  # G(s,a) = 0
        visited = set()

        for episode in range(1, nb_episodes + 1):
            state = self.env.reset()

            if state not in self.value_function.keys():
                self.value_function.update({state: {}})

            action = self._epsilon_greedy_action(learning_rate, episode + start_episode, state)
            done = False
            total_return = 0
            while not done:
                old_state = state  # save old state s
                old_action = action  # save old action a
                state, reward, done, obs = self.env.step(action)  # new state and action s', a'
                total_return = gamma * total_return + reward  # total return = G(s,a)
                if not self.first_visit or (old_state, old_action) not in visited:

                    # Collect cumul(s,a). Add to dict if this is the first occurrence
                    if old_state not in self.C.keys():
                        self.C.update({old_state: {}})

                    if old_action not in self.C[old_state].keys():
                        self.C[old_state].update({old_action: 1})
                        cumulative = 1
                    else:
                        self.C[old_state][old_action] += 1
                        cumulative = self.C[old_state][old_action]

                    # Collect V(s,a)
                    old_value = self.value_function.get(old_state, {}).get(old_action, 0)

                    # compute new value using incremental implementation (Sutton & Barto, Chapter 2)
                    new_value = old_value + (total_return - old_value) / cumulative

                    # Store V(s,a). Add s to dict if not present.
                    if old_state not in self.value_function.keys():
                        self.value_function.update({old_state: {}})

                    self.value_function[old_state].update({old_action: new_value})

                    # for first-visit methods
                    if self.first_visit:
                        visited.add((old_state, old_action))
            # There is no need to update the policy, as _epsilon_greedy_action always follows the correct formula

    def _epsilon_greedy_action(self, learning_rate: Callable[[int], float], nb_episodes: int, state):
        epsilon = learning_rate(nb_episodes)
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return self.predict(state)

    def predict(self, state):
        values_for_state = self.value_function.get(state, {})
        a_star = self._argmax_dict(values_for_state)
        return a_star

    def _argmax_dict(self, dc: dict):
        if len(dc.keys()) > 0:
            return max(dc.keys(), key=lambda key: dc.get(key, 0))
        else:  # in this case all values are zero, so argmax is just as good as a random sample
            return self.env.action_space.sample()

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump((self.value_function, self.C), f)
            f.close()

    @staticmethod
    def load(filename: str) -> Algorithm:
        with open(filename, 'rb') as f:
            value_func, cumul = pickle.load(f)
            f.close()
        return OnPolicyMCForTetris(value_function=value_func, cumul=cumul)
