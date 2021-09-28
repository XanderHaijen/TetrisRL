import concurrent.futures
import datetime
import os
import shutil
import sys


from Evaluation.train_and_test import train_and_test
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
from tetris_environment.tetris_env import TetrisEnv

args = []

path_to_scratch_dir = "/scratch/leuven/343/vsc34339/RLData"
path_to_data_dir = "/data/leuven/343/vsc34339/RLData"
#
path_to_scratch_dir = r"D:\Bibliotheken\Downloads\RLData"
path_to_data_dir = r"D:\Bibliotheken\OneDrive\Documenten\RLData"

# This file will train and test several combinations of alpha and gamma using a Sarsa(0) model
alpha_values = [0.05]
gamma_values = [0.9]
for alpha in alpha_values:
    for gamma in gamma_values:
        args.append(SarsaZeroAfterStates(TetrisEnv(type='fourer', render=False), alpha, gamma))


# trained simultaneously with concurrent.futures
def main(func_arg: SarsaZeroAfterStates) -> str:
    """

    :param func_arg: the model to be trained
    :return: string for reporting
    """

    # all with the same learning rate of epsilon
    def epsilon(x: int) -> float:
        return 0.01 if x < 5000 else 0.001

    # Unpack func_args
    model: SarsaZeroAfterStates = func_arg

    name_path = os.path.join(path_to_scratch_dir, f"{model.env.type}_alpha_{model.alpha}_gamma_{model.gamma}")
    data_path = os.path.join(name_path, "Data")
    model_dir = os.path.join(name_path, "StateValueModel")

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

    return f"StateValueModel gamma:{model.gamma} alpha:{model.alpha} done at {datetime.datetime.now()}"


for arg in args:
    result = main(arg)
    print(result)

# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.submit(main, func_arg) for func_arg in args]
#         for fs in concurrent.futures.as_completed(results):
#             print(fs.result())
