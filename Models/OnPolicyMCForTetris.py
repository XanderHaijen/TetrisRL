import pickle
import random
from typing import Callable
from Models.StateActionModel import StateValueModel
from tetris_environment.tetris_env import TetrisEnv


class OnPolicyMCForTetris(StateValueModel):
    def __init__(self, env: TetrisEnv, gamma: float, value_function: dict = None, Q: dict = None, C: dict = None,
                 first_visit: bool = True) -> None:
        """
        Initializes a trainable Monte Carlo model.
        The model uses on-policy first-visit or every-visit MC control for epsilon-soft policies. It does not
        require the assumption of exploring starts, but does require epsilon-soft policies.
        :param value_function: a dict of dicts for all state-action pairs. If no value function is provided,
        the values are initialized as zero for all state-action pairs
        :param Q: parameter in the learning process containing the average returns
        :param C: a kind of counter in the learning process.
        :param first_visit: specifies whether the model is first-visit or every-visit MC (Sutton & Barto, sec. 5.1)
        :param gamma: Importance sampling factor. 0 < gamma < 1
        """
        super().__init__(env)

        # the value function, C and Q are represented by a dict of dicts. State-action pairs are stored as
        # {state: {action: value}}. Non-visited state-action pairs are not stored and their
        # values are considered zero in the beginning.
        # The size of this dict is thus num_states. Each nested dict has max length _nb_actions for each state
        if C is None:
            C = {}
        if Q is None:
            Q = {}
        if value_function is None:
            value_function = {}

        self.value_function = value_function
        self.first_visit = first_visit
        self.Q = Q
        self.C = C
        self.gamma = gamma

    def train(self, learning_rate: Callable[[int], float], nb_episodes: int = 1000,
              start_episode: int = 0) -> None:
        """
        Trains the MC model using on-policy MC control for epsilon soft policies (Sutton & Barto, sec. 5.4)
        :ivar: 0 <= gamma <= 1
        :param learning_rate: value of epsilon. Must become zero only in the limit, otherwise convergence
        is not guaranteed
        :param nb_episodes: number of episodes to train the agent
        :param start_episode: if working with an already (partially) trained agent, provide the number of already
        trained episodes
        :return: None
        """
        if self.first_visit:  # first-visit MC control
            self.C = {}  # initialize self.C(s,a)

        for episode in range(1, nb_episodes + 1):
            # start the new episode
            state = self.env.reset()
            ext_state = (state, self.env.get_falling_piece())
            visited_pairs = set()  # set of every (s,a) visited in the episode

            # Take first action
            action = self._epsilon_greedy_action(learning_rate, episode + start_episode, ext_state)

            # play entire episode
            total_return = 0
            done = False
            while not done:
                old_ext_state = ext_state  # save old state s
                old_action = action  # save old action a
                state, reward, done, obs = self.env.step(action)  # take action a, observe s', R_(t+1)
                piece = self.env.get_falling_piece()

                if piece is not None:
                    ext_state = (state, piece)
                    total_return = self.gamma * total_return + reward
                    if not self.first_visit or (old_ext_state, old_action) not in visited_pairs:

                        # collect Q(s,a)
                        if old_ext_state not in self.Q.keys():
                            self.Q.update({old_ext_state: {}})
                        return_so_far = self.Q[old_ext_state].get(old_action, 0)

                        # Collect self.C(s,a) and update
                        if old_ext_state not in self.C.keys():
                            self.C.update({old_ext_state: {}})
                            cumulative = 1
                            self.C[old_ext_state].update({old_action: 1})
                        elif old_action not in self.C[old_ext_state].keys():
                            cumulative = 1
                            self.C[old_ext_state].update({old_action: 1})
                        else:
                            self.C[old_ext_state][old_action] += 1
                            cumulative = self.C[old_ext_state][old_action]

                        # compute new value for Q(s,a) and store in Q
                        return_so_far = return_so_far + (total_return - return_so_far) / cumulative
                        self.Q[old_ext_state].update({old_action: return_so_far})
                    visited_pairs.add((old_ext_state, old_action))
                    action = self._epsilon_greedy_action(learning_rate, episode + start_episode, ext_state)

                else:  # if piece is None, take no-action
                    action = self.env.no_move

            # After episode: update value function
            visited_states = {state for state, action in visited_pairs}
            for visited_state in visited_states:
                if visited_state not in self.value_function.keys():
                    self.value_function.update({visited_state: self.Q[visited_state]})
                else:
                    self.value_function[visited_state].update(self.Q[visited_state])

    def _epsilon_greedy_action(self, learning_rate: Callable[[int], float], nb_episodes: int, ext_state):
        epsilon = learning_rate(nb_episodes)
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return self.predict(ext_state)

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
            pickle.dump((self.value_function, self.gamma, self.C, self.Q, self.first_visit, self.env.type), f)
            f.close()

    @staticmethod
    def load(filename: str, rendering: bool = False) -> StateValueModel:
        with open(filename, 'rb') as f:
            value_func, gamma, C, Q, first_visit, size = pickle.load(f)
            f.close()
        env = TetrisEnv(type=size, render=rendering)
        return OnPolicyMCForTetris(env=env, gamma=gamma, value_function=value_func, C=C, Q=Q, first_visit=first_visit)

    def __str__(self):
        visit = "first-visit" if self.first_visit else "every-visit"
        return f"On-policy {visit} {self.env.type} MC model with gamma={self.gamma}"
