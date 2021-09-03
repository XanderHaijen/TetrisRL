import random
from abc import abstractmethod

from Algorithm import Algorithm
import numpy as np

class SarsaZero(Algorithm):
    """
    A Sarsa algorithm working with a state-action value function Q(s,a)
    """
    def __init__(self, env, alpha=1, gamma=1):
        """

        :param env: the environment in which to train the algorithm
        :param alpha: step-size-parameter in the update rule
        :param gamma: parameter in the update rule
        """
        self.alpha = alpha
        self.gamma = gamma
        self.env = env

        # the value function is represented by a dict of np arrays. State-action pairs are stored as
        # {state: np.array(nb_actions)}. Non-visited states are not stored and their
        # values are considered zero for all actions
        # The size of this dict is thus num_states. Each np array has length nb_actions for each state
        self.value_function = { }

    def epsilon_greedy_action(self, learning_rate, nb_episodes, state):
        """

        :param epsilon: a function of the number of episodes which goes towards zero at infinity
        :return: the action according to the epsilon greedy policy
        """
        epsilon = learning_rate(nb_episodes)
        if random.randrange(0,1) <= epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            action = self.predict(state)
            return action

    def train(self, learning_rate, nb_episodes=1000, start_episode=0) -> None:
        """
        Updates the value function according to the Sarsa algorithm (Sutton & Barto, page 155)
        :param learning_rate: = epsilon. A function of the number of episodes which goes towards zero at infinity
        :param nb_episodes: the duration of ´´the training session´´
        :param start_episode: zero in the beginning, greater than zero when training an already (partially)
        trained agent
        :return:
        """

        for episode in range(nb_episodes): # for each episode
            state = self.env.reset()

            if state not in self.value_function.keys():
                self.value_function.update({state: np.array(np.zeroes(self.nb_actions()))})

            action = self.epsilon_greedy_action(learning_rate, episode + start_episode, state)
            done = False
            while not done:
                old_state = state  # save old state s
                old_action = action  # save old action a
                state, reward, done, obs = self.env.step(action)  # new state and action s', a'
                action = self.epsilon_greedy_action(learning_rate, episode + start_episode, state)

                # update value function at Q(s,a)
                value_at_next_state = self.value_function[state][action]
                old_value = self.value_function[old_state][old_action]
                new_value = old_value + self.alpha * (reward + self.gamma * value_at_next_state - old_value)
                self.value_function[old_state][old_action] = new_value

    def nb_actions(self):
        return len(self.env.game_state.get_action_set())

    def predict(self, state):
        values_for_state = self.value_function.get(state, np.array(np.zeros(self.nb_actions())))
        a_star = np.argmax(values_for_state) # A_star is the optimal action in state A
        return a_star
