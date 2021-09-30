import pickle
import random
from typing import Callable, Tuple
from Models.AfterstateModel import AfterstateModel
from tetris_environment.tetris_env import TetrisEnv


class OnPolicyMCAfterstates(AfterstateModel):

    def __init__(self, env: TetrisEnv, gamma: float = 1, value_function: dict = None, Q: dict = None, C: dict = None,
                 first_visit: bool = True) -> None:
        """
        Initializes a trainable Monte Carlo model using an afterstate value function.
        The model uses on-policy first-visit or every-visit MC control for epsilon-soft policies. It does not
        require the assumption of exploring starts, but does require epsilon-soft policies.
        :param value_function: a dict for all states
        :param Q: parameter in the learning process containing the average returns
        :param C: a kind of counter in the learning process.
        :param first_visit: specifies whether the model is first-visit or every-visit MC (Sutton & Barto, sec. 5.1)
        :param gamma: Importance sampling factor.
        :ivar 0 < gamma < 1
        """
        super().__init__(env)

        # The value function, C and Q are represented by a dictionary. Non-visited states are not stored
        # and their values are initialized as zero. The max size of these dicts is thus num_states/
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
        :param learning_rate: value of epsilon. Must become zero only in the limit, otherwise convergence
        is not guaranteed
        :param nb_episodes: number of episodes to train the agent
        :param start_episode: if working with an already (partially) trained agent, provide the number of already
        trained episodes
        :return: None
        """
        if self.first_visit:  # first-visit MC control
            self.C = {}  # initialize self.C(s')

            for episode in range(1, nb_episodes + 1):
                # start the new episode
                state = self.env.reset()
                visited_afterstates = set()  # set of every s' visited in the episode

                # Take first action
                done = False

                # play entire episode
                total_return = 0
                while not done:
                    _, actions = self._epsilon_greedy_actions(learning_rate, episode + start_episode)
                    reward = 0
                    for action in actions:
                        # take action a, observe R and s' until the piece has reached the bottom
                        state, extra_reward, done, obs = self.env.step(action)
                        reward += extra_reward
                    total_return = self.gamma * total_return + reward

                    if not self.first_visit or state not in visited_afterstates:
                        return_so_far = self.Q.get(state, 0)  # Collect Q(s')

                        # Collect C(s') and update
                        if state not in self.C.keys():
                            self.C.update({state: 1})
                            cumulative = 1
                        else:
                            self.C[state] += 1
                            cumulative = self.C.get(state)

                        # Compute new value for Q(s') and store in Q
                        return_so_far = return_so_far + (total_return - return_so_far) / cumulative
                        self.Q.update({state: return_so_far})
                        visited_afterstates.add(state)

                for visited_state in visited_afterstates:
                    self.value_function.update({visited_state: self.Q[visited_state]})

    def _epsilon_greedy_actions(self, learning_rate: Callable[[int], float], nb_episodes: int) -> Tuple[tuple, list]:
        epsilon = learning_rate(nb_episodes)
        if random.random() <= epsilon:
            return self._pick_random_actions()
        else:
            return self.predict()

    def predict(self) -> Tuple[tuple, list]:
        possible_placements = self.env.all_possible_placements()
        # all possible placements for the current piece
        # return is of the form (afterstate, action)
        if len(possible_placements) > 0:
            best_placement = max(possible_placements, key=lambda pl: self.value_function.get(pl[0], 0))
        else:
            # no piece, so no action. State is also irrelevant, so None value to reduce computation
            best_placement = None, [0, ]
        return best_placement

    def _pick_random_actions(self) -> Tuple[tuple, list]:
        possible_placements = self.env.all_possible_placements()
        if len(possible_placements) > 0:
            placement = random.choice(possible_placements)
        else:
            # no piece, so no action. State is also irrelevant, so None value to reduce computation
            placement = None, [0, ]
        return placement

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump((self.value_function, self.C, self.Q, self.first_visit, self.env.type), f)
            f.close()

    @staticmethod
    def _load_file(filename: str):
        with open(filename, 'rb') as f:
            value_func, C, Q, first_visit, size = pickle.load(f)
            f.close()
        return value_func, C, Q, first_visit, size

    def load(self, filename: str, rendering: bool = False) -> AfterstateModel:
        value_func, C, Q, first_visit, size = self._load_file(filename)
        return OnPolicyMCAfterstates(env=TetrisEnv(type=size, render=rendering),
                                     value_function=value_func, C=C, Q=Q, first_visit=first_visit)
