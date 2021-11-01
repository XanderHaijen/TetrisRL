from Models.SarsaLambdaForTetris import SarsaLambdaForTetris
from Evaluation.train_and_test import train_and_test
from tetris_environment.tetris_env import TetrisEnv

model = SarsaLambdaForTetris(TetrisEnv(type="fourer"), alpha=0.01, gamma=0.9, Lambda=1, traces="dutch")
model.train(lambda x: 0.01, 10, 0)
