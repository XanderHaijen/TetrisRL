import numpy as np
import gym
from gym import spaces
import tetris_engine as game

SCREEN_WIDTH, SCREEN_HEIGHT = 200, 400


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # open up a game state to communicate with emulator
        self.game_state = game.TetrisGame()
        self._action_set = self.game_state.get_action_set()
        self.action_space = spaces.Discrete(len(self._action_set))
        # self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3))
        self.observation_space = spaces.Box(low=-3, high=3, shape=(9,), dtype=int)
        self.viewer = None

    def step(self, a):
        self._action_set = np.zeros([len(self._action_set)])
        self._action_set[a] = 1
        reward = 0.0
        state, reward, terminal, observations = self.game_state.frame_step(self._action_set)
        return state, reward, terminal, observations

    def get_image(self):
        return self.game_state.get_image()

    @property
    def n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        do_nothing = np.zeros(len(self._action_set))
        do_nothing[0] = 1
        # self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3))
        self.observation_space = spaces.Box(low=-3, high=3, shape=(9,), dtype=int)
        state, _, _, _ = self.game_state.frame_step(do_nothing)
        return state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def get_encoded_state(self, low: int = -3, high: int = 3):
        """
        Encodes the state space from the 10-by-20 (for a normal Tetris game) board to an integer array of size 9 by
        only noting the height differences between adjacent columns. If this difference is greater than :param high or
        smaller than :param low, the value is equal to :param high or :param low respectively.

        :param low: the lowest number which can be stored
        :param high: the highest number which can be stored
        :return: a numpy array of size board_width - 1 containing h_(i+1)-h_i in the i'th spot.
        """
        board_width = self.game_state.get_board_width()
        state = np.zeros(board_width - 1, dtype=int)
        for i in range(board_width - 1):
            height_diff = self.game_state.get_column_height(i + 1) - self.game_state.get_column_height(i)
            if height_diff < low:
                height_diff = low
            elif height_diff > high:
                height_diff = high
            state[i] = height_diff

        return state
