# Modified from Tetromino by lusob luis@sobrecueva.com
# http://lusob.com
# Released under a "Simplified BSD" license

# Modified to work for this project

from tetris_environment.tetris_engine import *
import pygame
import os


class RenderingTetrisGame(UnrenderedTetrisGame):
    def __init__(self, type: str, board=None):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT, WINDOWWIDTH
        WINDOWWIDTH = BOXSIZE * 4 if type == 'fourer' else BOXSIZE * 10
        pygame.init()
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
        pygame.display.iconify()
        pygame.display.set_caption('Tetromino')
        super().__init__(type, board)

    def reinit(self):
        """
        Re-initializes the board to an empty board
        """
        super().reinit()

        pygame.display.update()

    def frame_step(self, input):
        _, reward, terminal, data = super().frame_step(input)

        if not terminal:
            DISPLAYSURF.fill(BGCOLOR)
            self.draw_board()
            if self.fallingPiece != None:
                self.draw_piece(self.fallingPiece)
            pygame.display.update()

        return None, reward, terminal, data

    def get_image(self):
        image_data = pygame.surfarray.array3d(pygame.transform.rotate(pygame.display.get_surface(), 90))
        return image_data

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
                         (XMARGIN - 3, TOPMARGIN - 7, (self.board_width * BOXSIZE) + 8,
                          (BOARDHEIGHT * BOXSIZE) + 8), 5)

        # fill the background of the self.board
        pygame.draw.rect(DISPLAYSURF, BGCOLOR,
                         (XMARGIN, TOPMARGIN, BOXSIZE * self.board_width, BOXSIZE * BOARDHEIGHT))
        # draw the individual boxes on the self.board
        for x in range(self.board_width):
            for y in range(BOARDHEIGHT):
                self.draw_box(x, y, self.board[x][y])

    def draw_status(self):
        # draw the self.score text
        scoreSurf = BASICFONT.render('self.score: %s' % self.score, True, TEXTCOLOR)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (self.board_width * BOXSIZE - 150, 20)
        DISPLAYSURF.blit(scoreSurf, scoreRect)

        # draw the self.level text
        levelSurf = BASICFONT.render('self.level: %s' % self.level, True, TEXTCOLOR)
        levelRect = levelSurf.get_rect()
        levelRect.topleft = (self.board_width * BOXSIZE - 150, 50)
        DISPLAYSURF.blit(levelSurf, levelRect)

    def draw_piece(self, piece, pixelx=None, pixely=None):
        shapeToDraw = self.pieces[piece['shape']][piece['rotation']]
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
        nextRect.topleft = (self.get_board_width() * BOXSIZE - 120, 80)
        DISPLAYSURF.blit(nextSurf, nextRect)
        # draw the "next" piece
        self.draw_piece(self.nextPiece, pixelx=self.get_board_width() * BOXSIZE - 120, pixely=100)
