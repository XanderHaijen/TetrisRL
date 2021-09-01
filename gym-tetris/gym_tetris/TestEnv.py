import os
import time

from stable_baselines3 import A2C
from tetris_env import TetrisEnv

model_path = os.path.join(r"C:\Users\xande\Documents\TetrisNES\gym-tetris\gym_tetris","Training", "Model")
env = TetrisEnv()
model = A2C("MlpPolicy", env, verbose=1)

# # for learning only: very, very slow
# model.learn(total_timesteps=1000)
# model.save(model_path)
# del model

model.load(model_path)

NB_EPISODES = 2
for episode in range(1, NB_EPISODES + 1):
    observations = env.reset()
    done = False
    score = 0

    j = 0
    while not done:
        j += 1
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Reward = {reward}")
        time.sleep(0.1)
        reward += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
