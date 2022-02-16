import concurrent.futures
import datetime
import os
import shutil
import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.OnPolicyMCForTetris import OnPolicyMCForTetris
from tetris_environment.tetris_env import TetrisEnv

path_to_data_dir = "/scratch/leuven/343/vsc34339/RLData/MonteCarloExFourer"
# path_to_data_dir = r"C:\Users\xande\Downloads"

# This upper_dir will train several Monte Carlo agents using different values for gamma
gamma_values = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975]
visit = [True, False]
args = list()
for gamma in gamma_values:
    for first_visit in visit:
        args.append(OnPolicyMCForTetris(TetrisEnv(type='fourer', render=False),
                                          gamma=gamma, first_visit=first_visit))


def main(model: OnPolicyMCForTetris) -> str:

    def epsilon(x: int) -> float:
        return 0.001

    name_path = os.path.join(path_to_data_dir, f"{model}")
    data_path = os.path.join(name_path, "Data")
    model_dir = os.path.join(name_path, "Model")

    if os.path.isdir(name_path):
        shutil.rmtree(name_path)

    os.mkdir(name_path)
    os.mkdir(data_path)
    os.mkdir(model_dir)

    model_path = os.path.join(model_dir, "model.pickle")

    train_and_test(model,
                   epsilon,
                   model_path,
                   data_path,
                   10000, 20, 1000)

    return f"{model} done at {datetime.datetime.now()}."


# for arg in args:
#     result = main(arg)
#     print(result)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(main, args):
            print(result)
