import os
import random
import shutil
import time
from Evaluation.Evaluate_policy import evaluate_policy_afterstates
from Evaluation.Render_policy import render_policy_afterstates
from tetris_environment.tetris_env import TetrisEnv
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
from Models.OnPolicyMCAfterstates import OnPolicyMCAfterstates

models_dir = "scratch/leuven/343/vsc34339/RLData/fourer_sarsa"
target_dir = "data/leuven/343/vsc34339/RLData"
text_file = os.path.join(target_dir, "results.txt")
f = open(text_file, "w+")
for model_dir in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_dir, "Model", "model.pickle")
    model = SarsaZeroAfterStates.load(model_path)
    metrics = evaluate_policy_afterstates(model, model.env, 2000)
    results_dir = os.path.join(target_dir, "Results")

    f.write(f"\n \n \n For {model}: \n \n ")
    f.write("Mean and quantiles: \n")
    f.write(str(metrics.mean()))
    f.write("\n")
    f.write(str(metrics.quantile([0.25, 0.5, 0.75])))
    f.write("\n")
    f.write("Variance: \n")
    f.write(str(metrics.var()))
    f.write("\n Maximum and minimum \n ")
    f.write(str(metrics.max()))
    f.write(str(metrics.min()))
    f.write("\n ------------------------------------------------------------"
            " \n ------------------------------------------------------------ \n ")
    metrics.to_csv(os.path.join(results_dir, f"{model}.csv"))

f.write("The end")

# model = OnPolicyMCAfterstates(gamma=0.9, first_visit=False, env=TetrisEnv(type='fourer', render=False))
# model.train(lambda x: 0.01, 10)
# model.save(r"C:\Users\xande\Downloads\mc.pickle")
# print(model)
# model = SarsaZeroAfterStates.load(r'C:\Users\xande\Downloads\2nd_model.pickle', False)
# print("Trained:")
# metrics = evaluate_policy_afterstates(model, model.env, 500)
# print(metrics)
# print("Quantiles")
# print(metrics.quantile([0.25, 0.5, 0.75]))
# print("Mean")
# print(metrics.mean())
#
# model.env = TetrisEnv('fourer', True)
# render_policy_afterstates(model, model.env, 10)

# print("Trained")
# model = OnPolicyMCAfterstates.load(filename=r"C:\Users\xande\Downloads\2nd_model.pickle",
#                                   rendering=False)
#
# metrics = evaluate_policy_afterstates(model, model.env, 500)
# print(metrics)
# print("Max")
# print(metrics.max)
# print("Quantiles")
# print(metrics.quantile([0.25, 0.5, 0.75]))
# print("Mean")
# print(metrics.mean())
#
# model.env = TetrisEnv(type=model.env.type, render=True)
# render_policy_afterstates(model, model.env, 10)