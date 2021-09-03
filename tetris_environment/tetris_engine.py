# Modified from Tetromino by lusob luis@sobrecueva.com
# http://lusob.com
# Released under a "Simplified BSD" license

import random
import time
import pygame

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
    def __init__(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
        pygame.display.iconify()
        pygame.display.set_caption('Tetromino')

        # DEBUG
        self.total_lines = 0

        # For calculation of reward
        self.avg_height = 0
        self.holes = 0
        self.bumpiness = 0

        # setup variables for the start of the game
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

        self.board_width = BOARDWIDTH
        self.board_height = BOARDHEIGHT

        self.frame_step([1, 0, 0, 0, 0, 0])

        pygame.display.update()

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

        pygame.display.update()

    @property
    def get_board_width(self):
        return self.board_width

    @property
    def get_board_height(self):
        return self.board_height

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
                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                terminal = True

                self.reinit()
                reward = -10  # penalty for game over
                data = {"score": self.score, "lines_cleared": self.total_lines, "new_piece": new_piece}
                return image_data, reward, terminal, data  # can't fit a new piece on the self.board, so game over

        # moving the piece sideways
        if (input[1] == 1) and self.is_valid_position(adjX=-1):
            self.fallingPiece['x'] -= 1
            self.movingLeft = True
            self.movingRight = False
            self.lastMoveSidewaysTime = time.time()

        elif (input[3] == 1) and self.is_valid_position(adjX=1):
            self.fallingPiece['x'] += 1
            self.movingRight = True
            self.movingLeft = False
            self.lastMoveSidewaysTime = time.time()

        # rotating the piece (if there is room to rotate)
        elif (input[2] == 1):
            self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] + 1) % len(
                PIECES[self.fallingPiece['shape']])
            if not self.is_valid_position():
                self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] - 1) % len(
                    PIECES[self.fallingPiece['shape']])

        elif (input[5] == 1):  # rotate the other direction
            self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] - 1) % len(
                PIECES[self.fallingPiece['shape']])
            if not self.is_valid_position():
                self.fallingPiece['rotation'] = (self.fallingPiece['rotation'] + 1) % len(
                    PIECES[self.fallingPiece['shape']])

        # move the current piece all the way down
        elif (input[4] == 1):
            self.movingDown = False
            self.movingLeft = False
            self.movingRight = False
            for i in range(1, BOARDHEIGHT):
                if not self.is_valid_position(adjY=i):
                    break
            self.fallingPiece['y'] += i - 1

        # handle moving the piece because of user input
        if (self.movingLeft or self.movingRight):
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

        data = {"score": self.score, "lines_cleared": self.total_lines, "new_piece": new_piece}

        # drawing everything on the screen
        DISPLAYSURF.fill(BGCOLOR)
        self.draw_board()
        # self.drawStatus()
        # self.drawNextPiece()
        if self.fallingPiece != None:
            self.draw_piece(self.fallingPiece)

        pygame.display.update()
        #
        # if cleared > 0:
        #     reward = 100 * cleared

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        reward = self.get_reward()
        return image_data, reward, terminal, data

    def get_image(self):
        image_data = pygame.surfarray.array3d(pygame.transform.rotate(pygame.display.get_surface(), 90))
        return image_data

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

    def get_holes_diff(self):
        """
        :return: holes^t-holes^(t+1)
        """
        nb_holes = 0

        for col in range(BOARDWIDTH):
            i = 0
            while i < BOARDHEIGHT and self.board[col][i] == ".":
                i += 1
            nb_holes += len([x for x in self.board[col][i+1:] if x == "."])
            # nb_holes += max(len([x for x in self.board[col][i+1:] if x == "."]), 3)  # use to limit hole penalty

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

        for col in range(BOARDWIDTH):
            i = 0
            while i < BOARDHEIGHT and self.board[col][i] == ".":
                i += 1
            min_ys.append(i)

        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            total_bumpiness += bumpiness

        bumpiness_diff = self.bumpiness - total_bumpiness
        self.bumpiness = total_bumpiness

        return bumpiness_diff

    def get_column_height(self, column_nb):
        for row in range(BOARDHEIGHT):
            if self.board[column_nb][row] != ".":
                return BOARDHEIGHT - row
        return 0
        # Based on:

        # for i in range(0, BOARDHEIGHT):
        #     blank_row = True
        #     for j in range(0, BOARDWIDTH):
        #         if self.board[j][i] != '.':
        #             num_blocks += 1
        #             blank_row = False
        #     if not blank_row and stack_height is None:
        #         stack_height = BOARDHEIGHT - i

    def get_reward(self):
        """
        Reward consists out of 3 parts:
            1) the difference in average height
            2) the difference in holes
            3) the difference in quadratic unevenness

        :ivar: ALPHA, BETA, GAMMA greater than zero
        :return: r_(t+1)=α(h_avg^t-h_avg^(t+1) )+β(holes^t-holes^(t+1) )+γ(U^t-U^(t+1) ) where α,β,γ>0
        """
        ALPHA = 1
        BETA = 1
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

    def is_valid_position(self, adjX=0, adjY=0):
        # Return True if the piece is within the self.board and not colliding
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                is_above_board = y + self.fallingPiece['y'] + adjY < 0
                if is_above_board or PIECES[self.fallingPiece['shape']][self.fallingPiece['rotation']][y][x] == BLANK:
                    continue
                if not self.is_on_board(x + self.fallingPiece['x'] + adjX, y + self.fallingPiece['y'] + adjY):
                    return False
                if self.board[x + self.fallingPiece['x'] + adjX][y + self.fallingPiece['y'] + adjY] != BLANK:
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

    def draw_box(self, boxx, boxy, color, pixelx=None, pixely=None):
        # draw a single box (each tetromino piece has four boxes)
        # at xy coordinates on the self.board. Or, if pixelx & pixely
        # are specified, draw to the pixel coordinates stored in
        # pixelx & pixely (this is used for the "Next" piece).
        if color == BLANK:
            return
        if pixelx is None and pixely is None:
            pixelx, pixely = self.convert_to_pixel_coords(boxx, boxy)
        pygame.draw.rect(DISPLAYSURF, COLORS[color], (pixelx + 1, pixely + 1, BOXSIZE - 1, BOXSIZE - 1))
        pygame.draw.rect(DISPLAYSURF, LIGHTCOLORS[color], (pixelx + 1, pixely + 1, BOXSIZE - 4, BOXSIZE - 4))

    def draw_board(self):
        # draw the border around the self.board
        pygame.draw.rect(DISPLAYSURF, BORDERCOLOR,
                         (XMARGIN - 3, TOPMARGIN - 7, (BOARDWIDTH * BOXSIZE) + 8, (BOARDHEIGHT * BOXSIZE) + 8), 5)

        # fill the background of the self.board
        pygame.draw.rect(DISPLAYSURF, BGCOLOR, (XMARGIN, TOPMARGIN, BOXSIZE * BOARDWIDTH, BOXSIZE * BOARDHEIGHT))
        # draw the individual boxes on the self.board
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                self.draw_box(x, y, self.board[x][y])

    def draw_status(self):
        # draw the self.score text
        scoreSurf = BASICFONT.render('self.score: %s' % self.score, True, TEXTCOLOR)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (WINDOWWIDTH - 150, 20)
        DISPLAYSURF.blit(scoreSurf, scoreRect)

        # draw the self.level text
        levelSurf = BASICFONT.render('self.level: %s' % self.level, True, TEXTCOLOR)
        levelRect = levelSurf.get_rect()
        levelRect.topleft = (WINDOWWIDTH - 150, 50)
        DISPLAYSURF.blit(levelSurf, levelRect)

    def draw_piece(self, piece, pixelx=None, pixely=None):
        shapeToDraw = PIECES[piece['shape']][piece['rotation']]
        if pixelx == None and pixely == None:
            # if pixelx & pixely hasn't been specified, use the location stored in the piece data structure
            pixelx, pixely = self.convert_to_pixel_coords(piece['x'], piece['y'])

        # draw each of the boxes that make up the piece
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if shapeToDraw[y][x] != BLANK:
                    self.draw_box(None, None, piece['color'], pixelx + (x * BOXSIZE), pixely + (y * BOXSIZE))

    def draw_next_piece(self):
        # draw the "next" text
        nextSurf = BASICFONT.render('Next:', True, TEXTCOLOR)
        nextRect = nextSurf.get_rect()
        nextRect.topleft = (WINDOWWIDTH - 120, 80)
        DISPLAYSURF.blit(nextSurf, nextRect)
        # draw the "next" piece
        self.draw_piece(self.nextPiece, pixelx=WINDOWWIDTH - 120, pixely=100)
