import time

from Models.Model import Model
from gym import Env
from tetris_environment_rendered.tetris_env import TetrisEnv


def render_policy(model: Model, nb_episodes: int):
    env: Env = TetrisEnv()
    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            actions = model.predict(state)
            for action in actions:
                state, reward, done, data = env.step(action)
                env.render()
                time.sleep(0.1)
                print(reward)
