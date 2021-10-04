import random
import time
from Evaluation.Evaluate_policy import evaluate_policy_afterstates
from Evaluation.Render_policy import render_policy_afterstates
from tetris_environment.tetris_env import TetrisEnv
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
from Models.OnPolicyMCAfterstates import OnPolicyMCAfterstates

model = SarsaZeroAfterStates.load(r'C:\Users\xande\Downloads\2nd_model.pickle', False)
print("Trained:")
metrics = evaluate_policy_afterstates(model, model.env, 500)
print(metrics)
print("Quantiles")
print(metrics.quantile([0.25, 0.5, 0.75]))
print("Mean")
print(metrics.mean())

model.env = TetrisEnv('fourer', True)
render_policy_afterstates(model, model.env, 10)

# print("Trained")
# model = OnPolicyMCAfterstates.load(filename=r"C:\Users\xande\Downloads\MC_model.pickle",
#                                   rendering=False)
# metrics = evaluate_policy_afterstates(model, model.env, 500)
# print(metrics)
# print("Quantiles")
# print(metrics.quantile([0.25, 0.5, 0.75]))
# print("Mean")
# print(metrics.mean())

