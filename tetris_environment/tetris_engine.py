# Modified from Tetromino by lusob luis@sobrecueva.com
# http://lusob.com
# Released under a "Simplified BSD" license

# Modified to work for this project

import random
import time
from math import sqrt

FPS = 25
BOXSIZE = 20
BOARDWIDTH = 10
BOARDHEIGHT = 20
WINDOWWIDTH = BOXSIZE * BOARDWIDTH
WINDOWHEIGHT = BOXSIZE * BOARDHEIGHT
BLANK = '.'

MOVESIDEWAYSFREQ = 0.15
MOVEDOWNFREQ = 0.1

XMARGIN = 0
TOPMARGIN = 0

#         R    G    B
WHITE = (255, 255, 255)
GRAY = (185, 185, 185)
BLACK = (0, 0, 0)
RED = (155, 0, 0)
LIGHTRED = (175, 20, 20)
GREEN = (0, 155, 0)
LIGHTGREEN = (20, 175, 20)
BLUE = (0, 0, 155)
LIGHTBLUE = (20, 20, 175)
YELLOW = (155, 155, 0)
LIGHTYELLOW = (175, 175, 20)

BORDERCOLOR = BLUE
BGCOLOR = BLACK
TEXTCOLOR = WHITE
TEXTSHADOWCOLOR = GRAY
COLORS = (BLUE, GREEN, RED, YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)
assert len(COLORS) == len(LIGHTCOLORS)  # each color must have light color

TEMPLATEWIDTH = 5
TEMPLATEHEIGHT = 5

S_SHAPE_TEMPLATE = [['..OO.',
                     '.OO..',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..OO.',
                     '...O.',
                     '.....',
                     '.....']]

Z_SHAPE_TEMPLATE = [['.OO..',
                     '..OO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '.OO..',
                     '.O...',
                     '.....',
                     '.....']]

I_SHAPE_TEMPLATE = [['..O..',
                     '..O..',
                     '..O..',
                     '..O..',
                     '.....'],
                    ['OOOO.',
                     '.....',
                     '.....',
                     '.....',
                     '.....']]

O_SHAPE_TEMPLATE = [['.OO..',
                     '.OO..',
                     '.....',
                     '.....',
                     '.....']]

J_SHAPE_TEMPLATE = [['.O...',
                     '.OOO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..OO.',
                     '..O..',
                     '..O..',
                     '.....',
                     '.....'],
                    ['.OOO.',
                     '...O.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..O..',
                     '.OO..',
                     '.....',
                     '.....']]

L_SHAPE_TEMPLATE = [['...O.',
                     '.OOO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..O..',
                     '..OO.',
                     '.....',
                     '.....'],
                    ['.OOO.',
                     '.O...',
                     '.....',
                     '.....',
                     '.....'],
                    ['.OO..',
                     '..O..',
                     '..O..',
                     '.....',
                     '.....']]

T_SHAPE_TEMPLATE = [['..O..',
                     '.OOO.',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '..OO.',
                     '..O..',
                     '.....',
                     '.....'],
                    ['.OOO.',
                     '..O..',
                     '.....',
                     '.....',
                     '.....'],
                    ['..O..',
                     '.OO..',
                     '..O..',
                     '.....',
                     '.....']]

PIECES = {'S': S_SHAPE_TEMPLATE,
          'Z': Z_SHAPE_TEMPLATE,
          'J': J_SHAPE_TEMPLATE,
          'L': L_SHAPE_TEMPLATE,
          'I': I_SHAPE_TEMPLATE,
          'O': O_SHAPE_TEMPLATE,
          'T': T_SHAPE_TEMPLATE}


class TetrisGame:
    def __init__(self, board=None):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT

        # DEBUG
        self.total_lines = 0

        # For calculation of reward
        self.avg_height = 0
        self.holes = 0
        self.bumpiness = 0

        # setup variables for the start of the game
        self.board = self.get_blank_board() if board is None else board
        self.lastMoveDownTime = time.time()
        self.lastMoveSidewaysTime = time.time()
        self.lastFallTime = time.time()
        self.movingDown = False  # note: there is no movingUp variable
        self.movingLeft = False
        self.movingRight = False
        self.score = 0
        self.lines = 0
        self.height = 0
        self.level, self.fallFreq = self.calculate_level_and_fall_freq()

        self.fallingPiece = self.get_new_piece()
        self.nextPiece = self.get_new_piece()

        self.board_width = BOARDWIDTH
        self.board_height = BOARDHEIGHT

        self.frame_step([1, 0, 0, 0, 0, 0])
        self.pieces = PIECES

    def reinit(self):
        """
        Re-initializes the board to an empty board
        """

        self.avg_height = 0
        self.holes = 0
        self.bumpiness = 0

        self.board = self.get_blank_board()
        self.lastMoveDownTime = time.time()
        self.lastMoveSidewaysTime = time.time()
        self.lastFallTime = time.time()
        self.movingDown = False  # note: there is no movingUp variable
        self.movingLeft = False
        self.movingRight = False
        self.score = 0
        self.lines = 0
        self.height = 0
        self.level, self.fallFreq = self.calculate_level_and_fall_freq()

        self.fallingPiece = self.get_new_piece()
        self.nextPiece = self.get_new_piece()

        self.frame_step([1, 0, 0, 0, 0, 0])

    @property
    def get_board_width(self):
        return self.board_width

    @property
    def get_board_height(self):
        return self.board_height

    def move_left(self):
        """
        Moves the piece left if possible. If impossible, function doesn't do anything.
        :return:
        """
        if self.is_valid_position(adjX=-1):
            self.fallingPiece['x'] -= 1
            self.movingLeft = True
            self.movingRight = False
            self.lastMoveSidewaysTime = time.time()

    def move_right(self):
        """
        Move the piece right if possible. If impossible, function doesn't do anything
        :return:
        """
        if self.is_valid_position(adjX=1):
            self.fallingPiece['x'] += 1
            self.movingRight = True
            self.movingLeft = False
            self.lastMoveSidewaysTime = time.time()

    def rotate_clockwise(self):
        self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] + 1) % len(
            PIECES[self.fallingPiece['shape']])
        if not self.is_valid_position():
            self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] - 1) % len(
                PIECES[self.fallingPiece['shape']])

    def rotate_counterclockwise(self):
        self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] - 1) % len(
            PIECES[self.fallingPiece['shape']])
        if not self.is_valid_position():
            self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] + 1) % len(
                PIECES[self.fallingPiece['shape']])

    def move_down(self):
        """
        Move the piece all the way to the bottom.
        :return:
        """
        self.movingDown = False
        self.movingLeft = False
        self.movingRight = False
        for i in range(1, BOARDHEIGHT):
            if not self.is_valid_position(adjY=i):
                break
        self.fallingPiece['y'] += i - 1

    def frame_step(self, input):
        self.movingLeft = False
        self.movingRight = False

        new_piece = False
        terminal = False

        # none is 100000, left is 010000, up is 001000, right is 000100, space is 000010, q is 000001
        if self.fallingPiece == None:
            # No falling piece in play, so start a new piece at the top
            new_piece = True
            self.fallingPiece = self.nextPiece
            self.nextPiece = self.get_new_piece()
            self.lastFallTime = time.time()  # reset self.lastFallTime

            if not self.is_valid_position():
                terminal = True

                self.reinit()
                reward = -10  # penalty for game over
                data = {"score": self.score, "lines_cleared": self.total_lines, "new_piece": new_piece}
                return None, reward, terminal, data  # can't fit a new piece on the self.board, so game over

        # moving the piece sideways
        if (input[1] == 1) and self.is_valid_position(adjX=-1):
            self.move_left()
        elif (input[3] == 1) and self.is_valid_position(adjX=1):
            self.move_right()
        # rotating the piece (if there is room to rotate)
        elif input[2] == 1:
            self.rotate_clockwise()
        elif input[5] == 1:  # rotate the other direction
            self.rotate_counterclockwise()
        # move the current piece all the way down
        elif input[4] == 1:
            self.move_down()

        # handle moving the piece because of user input
        if self.movingLeft or self.movingRight:
            if self.movingLeft and self.is_valid_position(adjX=-1):
                self.fallingPiece['x'] -= 1
            elif self.movingRight and self.is_valid_position(adjX=1):
                self.fallingPiece['x'] += 1
            self.lastMoveSidewaysTime = time.time()

        if self.movingDown:
            self.fallingPiece['y'] += 1
            self.lastMoveDownTime = time.time()

        # let the piece fall if it is time to fall
        # see if the piece has landed
        cleared = 0
        if not self.is_valid_position(adjY=1):
            # falling piece has landed, set it on the self.board
            self.add_to_board()

            cleared = self.remove_complete_lines()
            if cleared > 0:
                if cleared == 1:
                    self.score += 40 * self.level
                elif cleared == 2:
                    self.score += 100 * self.level
                elif cleared == 3:
                    self.score += 300 * self.level
                elif cleared == 4:
                    self.score += 1200 * self.level

            self.score += self.fallingPiece['y']

            self.lines += cleared
            self.total_lines += cleared

            self.height = self.get_height()

            self.level, self.fallFreq = self.calculate_level_and_fall_freq()
            self.fallingPiece = None

        else:
            # piece did not land, just move the piece down
            self.fallingPiece['y'] += 1

        data = {"score": self.score, "lines_cleared": cleared, "new_piece": new_piece}
        reward = self.get_reward()
        return None, reward, terminal, data


    @staticmethod
    def get_action_set():
        return range(6)

    def get_height(self):
        stack_height = 0
        for i in range(0, BOARDHEIGHT):
            blank_row = True
            for j in range(0, BOARDWIDTH):
                if self.board[j][i] != '.':
                    blank_row = False
            if not blank_row:
                stack_height = BOARDHEIGHT - i
                break

        return stack_height

    @property
    def get_score(self):
        return self.score

    def is_surrounded(self, col_nb, row_nb) -> int:
        pass

    def get_holes_diff(self):
        """
        A hole is defined as a cell or region of cells of the Tetris board which is surrounded by a combination of filled cells
        and the floor/sides.
        :return: holes^t-holes^(t+1)
        """
        # MAX_HOLE_SIZE = 3
        # nb_holes = 0
        # checked_cells = set()
        # for col in range(BOARDWIDTH):
        #     for row in range(BOARDHEIGHT):
        #         nb_holes += min(self.is_surrounded(col, row), MAX_HOLE_SIZE)
        #
        #
        nb_holes = 0

        for col in range(BOARDWIDTH):
            i = 0
            while i < BOARDHEIGHT and self.board[col][i] == ".":
                i += 1
            # nb_holes += len([x for x in self.board[col][i+1:] if x == "."])
            nb_holes += min(len([x for x in self.board[col][i + 1:] if x == "."]), 3)  # use to limit hole penalty

        nb_holes_diff = self.holes - nb_holes
        self.holes = nb_holes

        return nb_holes_diff

    def get_avg_height_diff(self):
        """
        :return: h_avg^t-h_avg^(t+1)
        """
        old_height = self.avg_height
        total_height = 0
        for i in range(BOARDWIDTH):
            total_height += self.get_column_height(i)

        avg_height_new = total_height / BOARDWIDTH
        avg_height_diff = old_height - avg_height_new
        self.avg_height = avg_height_new

        return avg_height_diff

    def get_bumpiness_diff(self):
        """
        :return: quadratic unevenness or bumpiness: U_q^t-U_q^(t+1)
        """
        total_bumpiness = 0
        min_ys = []

        total_bumpiness = 0
        for col in range(BOARDWIDTH - 1):
            col_difference = pow(self.get_column_height(col) - self.get_column_height(col + 1), 2)
            total_bumpiness += col_difference

        total_bumpiness = sqrt(total_bumpiness)

        bumpiness_diff = self.bumpiness - total_bumpiness
        self.bumpiness = total_bumpiness

        return bumpiness_diff

    def get_column_height(self, column_nb):
        for row in range(BOARDHEIGHT):
            if self.board[column_nb][row] != ".":
                return BOARDHEIGHT - row
        return 0

    def get_reward(self) -> float:
        """
        Reward consists out of 3 parts:
            1) the difference in average height
            2) the difference in number of holes
            3) the difference in quadratic unevenness

        :ivar: ALPHA, BETA, GAMMA greater than or equal to zero
        :return: r_(t+1)=α(h_avg^t-h_avg^(t+1) )+β(holes^t-holes^(t+1) )+γ(U^t-U^(t+1) ) where α,β,γ>0
        """
        # Parameters as used in [Thiam, Kessler and Schwenker]
        ALPHA = 5
        BETA = 16
        GAMMA = 1
        reward = ALPHA * self.get_avg_height_diff() + BETA * self.get_holes_diff() + GAMMA * self.get_bumpiness_diff()

        return reward

    def is_game_over(self):
        return self.fallingPiece is None and not self.is_valid_position()

    @staticmethod
    def make_text_objects(text, font, color):
        surf = font.render(text, True, color)
        return surf, surf.get_rect()

    def calculate_level_and_fall_freq(self):
        # Based on the self.score, return the self.level the player is on and
        # how many seconds pass until a falling piece falls one space.
        self.level = min(int(self.lines / 10) + 1, 10)
        self.fallFreq = 0.27 - (self.level * 0.02)
        return self.level, self.fallFreq

    @staticmethod
    def get_new_piece():
        # return a random new piece in a random rotation and color
        shape = random.choice(list(PIECES.keys()))
        newPiece = {'shape': shape,
                    'rotation': random.randint(0, len(PIECES[shape]) - 1),
                    'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                    'y': 0,  # start it above the self.board (i.e. less than 0)
                    'color': random.randint(0, len(COLORS) - 1)}
        return newPiece

    def add_to_board(self):
        # fill in the self.board based on piece's location, shape, and rotation
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y][x] != BLANK:
                    # noinspection PyTypeChecker
                    self.board[x + self.fallingPiece['x']][y + self.fallingPiece['y']] = self.fallingPiece['color']

    def get_blank_board(self):
        # create and return a new blank self.board data structure
        self.board = []
        for i in range(BOARDWIDTH):
            self.board.append([BLANK] * BOARDHEIGHT)
        return self.board

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < BOARDWIDTH and y < BOARDHEIGHT

    def is_valid_position(self, adjX=0, adjY=0, piece=None):
        """
        Return True if the piece is within the self.board and not colliding with any pieces other than
        the falling piece.
        :param adjX:
        :param adjY:
        :param piece:
        :return:
        """
        if piece is None:
            piece = self.fallingPiece
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                is_above_board = y + piece['y'] + adjY < 0
                if is_above_board or PIECES[piece['shape']][piece['rotation']][y][x] == BLANK:
                    continue
                if not self.is_on_board(x + piece['x'] + adjX, y + piece['y'] + adjY):
                    return False
                if self.board[x + piece['x'] + adjX][y + piece['y'] + adjY] != BLANK:
                    return False
        return True

    def is_complete_line(self, y):
        # Return True if the line filled with boxes with no gaps.
        for x in range(BOARDWIDTH):
            if self.board[x][y] == BLANK:
                return False
        return True

    def remove_complete_lines(self):
        # Remove any completed lines on the self.board, move everything above them down,
        # and return the number of complete lines.
        num_lines_removed = 0
        y = BOARDHEIGHT - 1  # start y at the bottom of the self.board
        while y >= 0:
            if self.is_complete_line(y):
                # Remove the line and pull boxes down by one line.
                for pullDownY in range(y, 0, -1):
                    for x in range(BOARDWIDTH):
                        self.board[x][pullDownY] = self.board[x][pullDownY - 1]
                # Set very top line to blank.
                for x in range(BOARDWIDTH):
                    self.board[x][0] = BLANK
                num_lines_removed += 1
                # Note on the next iteration of the loop, y is the same.
                # This is so that if the line that was pulled down is also
                # complete, it will be removed.
            else:
                y -= 1  # move on to check next row up
        return num_lines_removed

    @staticmethod
    def convert_to_pixel_coords(boxx, boxy):
        # Convert the given xy coordinates of the self.board to xy
        # coordinates of the location on the screen.
        return (XMARGIN + (boxx * BOXSIZE)), (TOPMARGIN + (boxy * BOXSIZE))
