from Models.OnPolicyMCForTetris import OnPolicyMCForTetris
from Evaluation.Evaluate_policy import *
from Evaluation.Render_policy import *
from tetris_environment.tetris_env import TetrisEnv
from Models.SarsaLambdaAfterstates import SarsaLambdaAfterstates
from Evaluation.train_and_test import train_and_test

env = TetrisEnv(type='fourer', render=True)
model = SarsaLambdaAfterstates(env, 0.8, 0.2, 0.8, "accumulating")

train_and_test(model,
               lambda x: 0.001,
               )