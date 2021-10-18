import pickle
from typing import Callable, Tuple
import random
from Models.AfterstateModel import AfterstateModel
from tetris_environment.tetris_env import TetrisEnv


class SarsaLambdaAfterstates(AfterstateModel):
    def __init__(self,
                 env: TetrisEnv,
                 Lambda: float, alpha: float, gamma: float,
                 traces: str,
                 value_function: dict = None, eligibility: dict = None):
        """
        Initializes a sarsa-control model with a Tetris environment
        :param Lambda: determines the amount of bootstrapping.
            NOTE: if lambda = 0, better use SarsaZeroForTetris
        :param alpha: step-size parameter in the update rule
        :param gamma: parameter in the update rule
        :param traces: Either accumulating, dutch, replacing. Specifies the update rule for eligibility traces
        :param value_function: a dict of dicts containing the value for each state-action pair. If none is provided
        it is initialized as Q(s,a) = 0 for all s, a
        """
        super().__init__(env)

        if eligibility is None:
            eligibility = {}
        if value_function is None:
            value_function = {}

        if Lambda == 0:
            print("WARNING: If lambda = 0, better use SarsaZeroAfterstates")

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
        Predicts the value function and updates the epsilon-greedy policy according to the Sarsa(lambda) model
        (Sutton & Barto, section 7.5) using an afterstate value function
        :param learning_rate: = epsilon. A function of the number of episodes which goes to zero in the limit
        :param nb_episodes: the duration of one ´´training session´´
        :param start_episode: zero in the beginnen, greater than zero when training an already partially trained agent
        :return: None
        """
        # NOTE: eligibility traces will reset to 0 when their value is less than MIN_ELEG
        MIN_ELEG = 0.01
        for episode in range(1, nb_episodes + 1):
            state = self.env.reset()
            afterstate, actions = self._epsilon_greedy_actions(learning_rate, episode + start_episode)

            done = False
            while not done:
                reward = 0
                for action in actions:
                    # take action a, observe R and s until the piece has reached the bottom
                    state, extra_reward, done, obs = self.env.step(action)
                    reward += extra_reward

                if state not in self.value_function.keys():
                    self.value_function.update({state: 0})

                # Determine next action and next state
                afterstate, actions = self._epsilon_greedy_actions(learning_rate, episode + start_episode)

                # collect V(s') and V(s)
                value_at_curr_state = self.value_function.get(state, 0)
                value_at_afterstate = self.value_function.get(afterstate, 0)

                # compute delta
                delta = reward + self.gamma * value_at_afterstate - value_at_curr_state

                # Update eligibility traces
                eleg = self.eligibility.get(state, 0)

                if self.traces == "accumulating":
                    eleg = eleg + 1
                elif self.traces == "dutch":
                    eleg = (1 - self.alpha) * eleg + 1
                elif self.traces == "replacing":
                    eleg = 1
                else:
                    raise RuntimeError

                if abs(eleg) < MIN_ELEG:
                    self.eligibility.pop(state)
                else:
                    self.eligibility.update({state: eleg})

                # Update Q and E for all s in S, a in A(s) and simultaneously reduce size of E(s) to save on memory

                # Q(s) <- Q(s) + alpha * delta * E(s)
                # E(s) <- lambda * gamma * E(s)
                states = {s for s in self.value_function.keys() if s in self.eligibility.keys()}
                for s in states:
                    self.value_function[s] += self.alpha * delta * self.eligibility.get(s, 0)
                    self.eligibility[s] *= (self.gamma * self.Lambda)
                    if abs(self.eligibility[s]) < MIN_ELEG:
                        self.eligibility.pop(s)

    def _epsilon_greedy_actions(self, learning_rate: Callable[[int], float], nb_episodes: int) -> Tuple[tuple, list]:
        """
        :param nb_episodes: how far into learning is the agent
        :param learning_rate: a function of the number of episodes which goes towards zero at infinity
        :return: the action according to the epsilon greedy policy, in the form of a tuple
                (afterstate, actions)
        """
        epsilon = learning_rate(nb_episodes)
        if random.random() <= epsilon:
            return self._pick_random_actions()
        else:
            return self.predict()

    @property
    def _nb_actions(self) -> int:
        return len(self.env.game_state.get_action_set())

    def predict(self) -> Tuple[tuple, list]:
        possible_placements = self.env.all_possible_placements()
        # possible_placements of form (state, action)
        if len(possible_placements) > 0:
            best_placement = max(possible_placements, key=lambda pl: self.value_function.get(pl[0], 0))
        else:
            best_placement = (self.env.get_encoded_state(), [0])  # no piece, so no action
        return best_placement

    def _pick_random_actions(self) -> Tuple[tuple, list]:
        possible_placements = self.env.all_possible_placements()
        if len(possible_placements) > 0:
            placement = random.choice(possible_placements)  # consists of afterstate and action
        else:
            placement = (self.env.get_encoded_state(), [0])  # no piece, so no action
        return placement

    @staticmethod
    def _load_file(filename: str) -> tuple:
        with open(filename, 'rb') as f:
            attributes = pickle.load(f)
            f.close()
        return attributes

    @staticmethod
    def load(filename: str, rendering: bool = False) -> AfterstateModel:
        gamma, alpha, Lambda, value_function, eligibility, traces, size = SarsaLambdaAfterstates._load_file(filename)
        env = TetrisEnv(type=size, render=rendering)
        return SarsaLambdaAfterstates(env, Lambda, alpha, gamma, traces, value_function, eligibility)

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump((self.gamma, self.alpha, self.Lambda,
                         self.value_function, self.eligibility, self.traces,
                         self.env.type),
                        f)
            f.close()

    def __str__(self):
        return f"{self.env.type} Sarsa Lambda afterstate model (alpha={self.alpha}, gamma={self.gamma}, " \
               f"lambda={self.Lambda}) with {self.traces} traces"
