import copy
import random
from typing import Tuple

import numpy as np
import gym
from gym import spaces
from tetris_environment.tetris_engine import UnrenderedTetrisGame
from tetris_environment.rendering_tetris_engine import RenderingTetrisGame

SCREEN_WIDTH, SCREEN_HEIGHT = 200, 400


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, type: str, render: bool = False, low: int = -3, high: int = 3):

        if type not in ("fourer", 'regular', 'extended fourer'):
            raise RuntimeError("Invalid Tetris type")

        # open up a game state to communicate with emulator
        if not render:
            self.game_state = UnrenderedTetrisGame(type)
            self.game_type = UnrenderedTetrisGame
        else:
            self.game_state = RenderingTetrisGame(type)
            self.game_type = RenderingTetrisGame
            # raise RuntimeError()

        self._action_set = self.game_state.get_action_set()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=low, high=high, shape=(9,), dtype=int)
        self.viewer = None
        self.type = type
        self.rendering = render

    def step(self, a: int) -> Tuple[tuple, float, bool, dict]:
        """
        Takes one step in the environment using the defined action a
        :param a: the action to take
        :return: a 4-tuple consisting of
                • the current encoded state (see get_encoded_state())
                • the reward received
                • whether or not the end of an epîsode has been reached
                • several observations with their own labels
                    - label "score" (int): the total score received during this step and this step alone
                    - label "new_piece (bool): is True when a new falling piece was added to the board
                    - label "lines_cleared" (int): the total amount of lines cleared during this step
        """
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

        if not self.rendering:
            return

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.game_state.get_image()
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
        state = [0 for _ in range(board_width - 1)]  # list of zeroes

        low = -3
        high = 3

        for i in range(board_width - 1):
            height_diff = board.get_column_height(i + 1) - board.get_column_height(i)
            if height_diff < low:
                height_diff = low
            elif height_diff > high:
                height_diff = high
            state[i] = height_diff

        return tuple(state)  # finally, convert to hashable type, i.e. tuple

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
            for width in range(-4, self.game_state.board_width + 4):
                shape = piece['shape']
                for rotation in range(len(self.game_state.pieces.get(shape))):
                    new_piece = copy.deepcopy(piece)
                    new_piece['x'] = width
                    new_piece['rotation'] = rotation
                    if self.game_state.is_valid_position(piece=new_piece):
                        all_possible_positions.insert(random.randint(0, len(all_possible_positions)), new_piece)

            all_possible_placements = []
            for piece in all_possible_positions:
                board = copy.deepcopy(self.game_state.board)
                new_game = self.game_type(board=board, type=self.type)
                new_game.fallingPiece = piece
                actions = self._get_actions(old_board=self.game_state, new_board=new_game)
                move_down = np.zeros(6)
                move_down[4] = 1
                new_game.frame_step(move_down)
                state = self.get_encoded_state(new_game)
                all_possible_placements.append((state, actions))

            return all_possible_placements
        else:
            return []

    @staticmethod
    def _get_actions(old_board, new_board):
        """
        Finds the actions required to position the falling piece on :param old_board in the same way as
        the falling piece on :param new_board
        :return: a list of all actions required (first rotations, then lateral movement). The last action added
                is the move_down action
        :raise AssertionError if the shapes of the falling pieces are different
        """
        # Define actions
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
        if curr_piece['x'] < target_piece['x']:
            for _ in range(target_piece['x'] - curr_piece['x']):
                actions.append(move_right)
        if curr_piece['x'] > target_piece['x']:
            for _ in range(curr_piece['x'] - target_piece['x']):
                actions.append(move_left)
        if len(actions) == 0:
            # piece placements are identical, so no actions are added.
            pass

        actions.append(move_down)

        return actions
