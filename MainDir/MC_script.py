import concurrent.futures
import datetime
import os
import shutil
import sys

# sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.OnPolicyMCAfterstates import OnPolicyMCAfterstates
from tetris_environment.tetris_env import TetrisEnv

# path_to_data_dir = "/data/leuven/343/vsc34339/RLData"
path_to_data_dir = r"C:\Users\xande\Downloads"

# This file will train several Monte Carlo agents using different values for gamma
gamma_values = [0.9]
args = list()
for gamma in gamma_values:
    args.append(OnPolicyMCAfterstates(TetrisEnv(type='fourer', render=False), gamma=gamma))


def main(model: OnPolicyMCAfterstates) -> str:

    def epsilon(nb_ep: int) -> float:
        return 0.001

    name_path = os.path.join(path_to_data_dir, f"{model.env.type}_MC_gamma_{model.gamma}")
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
                   10, 5, 10)

    return f"Afterstate {model.env.type} MC model gamma:{model.gamma} done at {datetime.datetime.now()}"


for arg in args:
    result = main(arg)
    print(result)
