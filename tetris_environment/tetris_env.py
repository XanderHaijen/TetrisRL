import copy

import numpy as np
import gym
from gym import spaces
from tetris_environment import tetris_engine as game
from tetris_environment.tetris_engine import TetrisGame

SCREEN_WIDTH, SCREEN_HEIGHT = 200, 400


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, low: int = -3, high: int = 3):
        # open up a game state to communicate with emulator
        self.game_state = game.TetrisGame()
        self._action_set = self.game_state.get_action_set()
        self.action_space = spaces.Discrete(len(self._action_set))
        # self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3))
        self.observation_space = spaces.Box(low=low, high=high, shape=(9,), dtype=int)
        self.viewer = None

    def step(self, a):
        self._action_set = np.zeros([len(self._action_set)])
        self._action_set[a] = 1
        _, reward, terminal, observations = self.game_state.frame_step(self._action_set)
        state = self.get_encoded_state()
        return state, reward, terminal, observations

    @property
    def n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def reset(self):
        do_nothing = np.zeros(len(self._action_set))
        do_nothing[0] = 1
        # self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3))
        self.observation_space = spaces.Box(low=-3, high=3, shape=(9,), dtype=int)
        self.game_state.frame_step(do_nothing)
        state = self.get_encoded_state()
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

    def get_encoded_state(self, board=None) -> tuple:
        """
        Encodes the state space from the 10-by-20 (for a normal Tetris game) board to an integer array of size 9 by
        only noting the height differences between adjacent columns. If this difference is greater than 3 or
        smaller than -3, the value is equal to 3 or -3 respectively.
        :param board: the Tetris board to be encoded. If none is provided, the board of the class is used
        :return: a tuple of size board_width - 1 containing h_(i+1)-h_i in the i'th spot.
        """
        if board is None:
            board = self.game_state

        board_width = board.board_width
        state = [0 for _ in range(board_width - 1)]  # tuple of zeroes

        low = -3
        high = 3

        for i in range(board_width - 1):
            height_diff = board.get_column_height(i + 1) - board.get_column_height(i)
            if height_diff < low:
                height_diff = low
            elif height_diff > high:
                height_diff = high
            state[i] = height_diff

        return tuple(state)

    def all_possible_placements(self):
        """
        Move the piece from all the way left to all the way right, and rotate it in all possible ways.
        Then return the state achieved by dropping the piece down and the first (of several) actions to
        take to achieve this state
        :return:
        """
        all_possible_positions = []

        piece = self.game_state.fallingPiece
        if piece is not None:
            for width in range(1, 11):
                shape = piece['shape']
                for rotation in range(len(self.game_state.pieces.get(shape))):
                    new_piece = copy.deepcopy(piece)
                    new_piece['x'] = width
                    new_piece['rotation'] = rotation
                    if self.game_state.is_valid_position(piece=new_piece):
                        all_possible_positions.append(new_piece)

            all_possible_placements = []
            for piece in all_possible_positions:
                board = copy.deepcopy(self.game_state.board)
                new_game = TetrisGame(board=board)
                new_game.fallingPiece = piece
                action = self._get_actions(old_board=self.game_state, new_board=new_game)
                move_down = np.zeros(6)
                move_down[4] = 1
                new_game.frame_step(move_down)
                state = self.get_encoded_state(new_game)
                all_possible_placements.append((state, action))

            del new_game
            return all_possible_placements
        else:
            return []

    @staticmethod
    def _get_actions(old_board, new_board):
        # Define actions
        no_move = 0
        move_left = 1
        move_right = 3
        move_down = 4
        rotate_clockwise = 5

        actions = []

        curr_piece = old_board.fallingPiece
        target_piece = new_board.fallingPiece
        assert curr_piece['shape'] == target_piece['shape']

        if curr_piece['rotation'] != target_piece['rotation']:
            for _ in range(abs(curr_piece["rotation"] - target_piece["rotation"])):
                actions.append(rotate_clockwise)
        elif curr_piece['x'] < target_piece['x']:
            for _ in range(target_piece['x'] - curr_piece['x']):
                actions.append(move_right)
        elif curr_piece['x'] > target_piece['x']:
            for _ in range(curr_piece['x'] - target_piece['x']):
                actions.append(move_left)
        else:  # piece placements are identical
            actions = [no_move]

        actions.append(move_down)

        return actions

    def get_image(self):
        pass
