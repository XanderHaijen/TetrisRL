import pickle
from typing import Callable
import random
from Models.Model import Model


class SarsaLambdaForTetris(Model):
    def __init__(self, Lambda: float, alpha: float, gamma: float, traces: str, value_function=None, eligibility=None):
        """
        Initializes a sarsa-control algorithm with a Tetris environment
        :param Lambda: determines the amount of bootstrapping.
            NOTE: if lambda = 0, better use SarsaZeroForTetris
        :param alpha: step-size parameter in the update rule
        :param gamma: parameter in the update rule
        :param traces: Either accumulating, dutch, replacing. Specifies the update rule for eligibility traces
        :param value_function: a dict of dicts containing the value for each state-action pair. If none is provided
        it is initialized as Q(s,a) = 0 for all s, a
        """
        super().__init__()

        if eligibility is None:
            eligibility = {}
        if value_function is None:
            value_function = {}

        if Lambda == 0:
            raise Warning("If lambda = 0, better use SarsaZeroForTetris")

        self.value_function = value_function

        self.Lambda = Lambda
        self.alpha = alpha
        self.gamma = gamma

        if traces not in ("accumulating", "dutch", "replacing"):
            raise RuntimeError("traces parameter is invalid")
        self.traces = traces

        self.eligibility = eligibility  # E(s,a) = 0 for all s,a

    def train(self, learning_rate: Callable[[int], float], nb_episodes: int = 1000, start_episode: int = 0) -> None:
        """
        Predicts the value function and updates the epsilon-greedy policy according to the Sarsa(lambda) algorithm
        (Sutton & Barto, section 7.5)
        :param learning_rate: = epsilon. A function of the number of episodes which goes to zero in the limit
        :param nb_episodes: the duration of one ´´training session´´
        :param start_episode: zero in the beginnen, greater than zero when training an already partially trained agent
        :return: None
        """
        # NOTE: eligibility traces will reset to 0 when their value is less than MIN_ELEG
        MIN_ELEG = 0.01
        for episode in range(1, nb_episodes + 1):
            state = self.env.reset()

            if state not in self.value_function.keys():
                self.value_function.update({state: {}})

            action = self._epsilon_greedy_action(learning_rate, episode + start_episode, state)
            done = False

            while not done:
                old_state = state
                old_action = action
                state, reward, done, obs = self.env.step(action)
                action = self._epsilon_greedy_action(learning_rate, episode + start_episode, state)  # observe R, s', a'

                # collect Q(s,a) and Q(s', a')
                old_value = self.value_function.get(old_state, {}).get(old_action, 0)
                value = self.value_function.get(state, {}).get(action, 0)

                # compute delta
                delta = reward + self.gamma * value - old_value

                # Update eligibility traces
                if old_state not in self.eligibility.keys():
                    self.eligibility.update({old_state: {}})
                eleg = self.eligibility[old_state].get(old_action, 0)

                if self.traces == "accumulating":
                    eleg = eleg + 1
                elif self.traces == "dutch":
                    eleg = (1 - self.alpha) * eleg + 1
                elif self.traces == "replacing":
                    eleg = 1
                else:
                    raise RuntimeError

                if eleg > MIN_ELEG:
                    if old_action in self.eligibility[old_state].keys():
                        self.eligibility[old_state].pop(old_action)
                    else:
                        pass
                else:
                    self.eligibility[old_state].update({old_action: eleg})

                # Update Q and E for all s in S, a in A(s) and simultaneously reduce size of E(s,a) to save on memory

                # Q(s,a) <- Q(s,a) + alpha * delta * E(s,a)
                # E(s,a) <- lambda * gamma * E(s,a)
                for s in self.value_function.keys():
                    for a in self.value_function[s].keys():
                        if a in self.eligibility.get(s, {}).keys():
                            self.value_function[s][a] += self.alpha * delta * self.eligibility[s][a]
                            self.eligibility[s][a] *= (self.gamma * self.Lambda)
                            if self.eligibility[s][a] < MIN_ELEG:
                                self.eligibility[s].pop(a)
                            if len(self.eligibility[a].keys()) == 0:
                                self.eligibility.pop(a)

    def _epsilon_greedy_action(self, learning_rate: Callable[[int], float], nb_episodes, state):
        """
        :param state: the state for which to choose the epsilon greedy action
        :param nb_episodes: how far into learning is the agent
        :param learning_rate: a function of the number of episodes which goes towards zero at infinity
        :return: the action according to the epsilon greedy policy
        """
        epsilon = learning_rate(nb_episodes)
        if random.random() <= epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            action = self.predict(state)
            return action

    def predict(self, state):
        values_for_state = self.value_function.get(state, {})
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

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump((self.gamma, self.alpha, self.Lambda, self.value_function, self.eligibility, self.traces), f)
            f.close()

    @staticmethod
    def load(filename: str) -> Model:
        with open(filename, 'rb') as f:
            gamma, alpha, Lambda, value_function, eligibility, traces = pickle.load(f)
            f.close()
        return SarsaLambdaForTetris(Lambda, alpha, gamma, traces, value_function, eligibility)