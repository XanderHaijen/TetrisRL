import random
import time
from Evaluation.Evaluate_policy import evaluate_policy_afterstates
from Evaluation.Render_policy import render_policy_afterstates
from tetris_environment.tetris_env import TetrisEnv
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates

model = SarsaZeroAfterStates(env=TetrisEnv(type='fourer', render=False))
render_policy_afterstates(model, TetrisEnv(type='fourer', render=True), 5)
