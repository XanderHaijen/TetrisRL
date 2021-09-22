import random
import time
from tetris_environment.tetris_env import TetrisEnv
game = TetrisEnv(type='fourer', render=True)

for _ in range(500):
    a = random.randint(0, 5)
    game.step(a)
    game.render()
    time.sleep(0.1)
