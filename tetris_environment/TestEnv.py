import os
import time
import pandas as pd
from stable_baselines3 import PPO
from tetris_env import TetrisEnv

model_path = os.path.join(r"C:\Users\xande\Documents\TetrisNES\gym-tetris\gym_tetris","Training", "Model")
env = TetrisEnv()
model = PPO("MlpPolicy", env, verbose=1)

# for learning only: very, very slow
model.learn(total_timesteps=1000)
model.save(model_path)
# del model

# model.load(model_path)

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
nb_episodes = 1000
env = TetrisEnv()


metrics = []
for episode in range(1, nb_episodes + 1):
    observations = env.reset()
    done = False
    nb_pieces = 0
    j = 0
    data = {}
    while not done:
        j += 1
        # env.render()  # only for viewing purposes
        action = env.action_space.sample()
        state, reward, done, data = env.step(action)
        if data["new_piece"]:
            nb_pieces += 1

    lines_cleared = data.get("lines_cleared", 0)
    score = data.get("score", 0)
    metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": lines_cleared, "Score": score})

metrics_df = pd.DataFrame.from_records(metrics)

print(metrics_df.var())
print(metrics_df.mean())
