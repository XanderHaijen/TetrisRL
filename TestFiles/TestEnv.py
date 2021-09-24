import random
import time
from Evaluation.Evaluate_policy import evaluate_policy_afterstates
from Evaluation.Render_policy import render_policy_afterstates
from tetris_environment.tetris_env import TetrisEnv
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
game = TetrisEnv(type='regular', render=True)

model = SarsaZeroAfterStates.load(r'D:\Bibliotheken\Downloads\2nd_model.pickle')
env = TetrisEnv(type='regular', render=True)
print(evaluate_policy_afterstates(model, env, 10))