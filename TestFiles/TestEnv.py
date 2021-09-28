import random
import time
from Evaluation.Evaluate_policy import evaluate_policy_afterstates
from Evaluation.Render_policy import render_policy_afterstates
from tetris_environment.tetris_env import TetrisEnv
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates

model = SarsaZeroAfterStates(env=TetrisEnv(type='fourer', render=False))
model.train(learning_rate=lambda x: 0.001, nb_episodes=100, start_episode=0)
render_policy_afterstates(model, TetrisEnv(type='fourer', render=True), 10)