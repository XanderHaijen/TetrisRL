import sys
import os
import shutil
from math import log
import concurrent.futures

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.SarsaLambdaForTetris import SarsaLambdaForTetris
from Models.SarsaZeroForTetris import SarsaZeroForTetris
from Models.OnPolicyMCForTetris import OnPolicyMCForTetris

args = []

path_to_scratch_dir = "/scratch/leuven/343/vsc34339/RLData"
path_to_data_dir = "/data/leuven/343/vsc34339/RLData"

path_to_scratch_dir = r"D:\Bibliotheken\Downloads\RLData"
path_to_data_dir = r"D:\Bibliotheken\OneDrive\Documenten\RLData"

# This file trains and evaluates several Sarsa and Monte Carlo-based reinforcement learning agents
# First model: Sarsa(0) with defined values for alpha and gamma
alpha = 0.1
gamma = 0.8
args.append((SarsaZeroForTetris(alpha, gamma), "SarsaZero"))

# Next models: Sarsa(lambda) for all entered values for lambda and specified traces
lambda_values = [0.1, 0.5, 0.9]
traces = "accumulating"
for var_lambda in lambda_values:
    args.append((SarsaLambdaForTetris(var_lambda, alpha, gamma, traces), f"Sarsa_{var_lambda}"))
# Next: first-visit MC for all entered values for gamma
# Next: every-visit MC for all entered values for gamma
mc_gamma_values = [1, 0.9, 0.8]
for mc_gamma in mc_gamma_values:
    args.append((OnPolicyMCForTetris(mc_gamma, first_visit=True), f"First_visit_mc_{mc_gamma}"))
    args.append((OnPolicyMCForTetris(mc_gamma, first_visit=False), f"Every_visit_mc_{mc_gamma}"))


# Trained simultaneously using concurrent.futures on following main function
def main(func_args: tuple) -> str:
    """

    :param func_args: is a tuple with structure (model, name)
    :return: string for reporting
    """

    # All models trained with the same learning rate of epsilon
    def epsilon(x):
        return 1 / (15 + log(x+1))

    # Unpack func_args
    model, name = func_args

    # Construct unique paths
    name_path = os.path.join(path_to_scratch_dir, name)
    data_path = os.path.join(name_path, "DataAndPlots")
    model_dir = os.path.join(name_path, "StateValueModel")

    # and make the necessary directories
    os.mkdir(name_path)
    os.mkdir(data_path)
    os.mkdir(model_dir)

    model_path = os.path.join(model_dir, "model.pickle")

    train_and_test(model,
                   epsilon,
                   model_path,
                   data_path,
                   10, 10, 10)

    return f"StateValueModel {name} done"


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(main, func_args) for func_args in args]
        for fs in concurrent.futures.as_completed(results):
            print(fs.result())

    # copy all data to more permanent $VSC_DATA
    shutil.copy(path_to_scratch_dir, path_to_data_dir)
