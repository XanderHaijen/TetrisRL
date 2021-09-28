import time
from Models.Model import Model
from tetris_environment.tetris_env import TetrisEnv


def render_policy_afterstates(model: Model, env: TetrisEnv, nb_episodes: int = 10) -> None:
    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        env.render()
        done = False
        while not done:
            actions = model.predict(state)
            for action in actions:
                state, reward, done, obs = env.step(action)
                time.sleep(0.1)
                print(int(reward))
                env.render()


def render_policy_state_action(model: Model, env: TetrisEnv, nb_episodes: int = 10) -> None:
    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        env.render()
        done = False
        while not done:
            action = model.predict(state)
            state, reward, done, obs = env.step(action)
            time.sleep(0.1)
            print(int(reward))
            env.render()