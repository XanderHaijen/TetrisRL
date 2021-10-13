import concurrent.futures
import datetime
import os
import shutil
import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.SarsaZeroForTetris import SarsaZeroForTetris
from tetris_environment.tetris_env import TetrisEnv

args = []

path_to_data_dir = "/scratch/leuven/343/vsc34339/RLData/StateValue Sarsa"
# path_to_data_dir = r'C:\Users\xande\Downloads'


# This file will train and test several combinations of alpha and gamma using a Sarsa(0) model
alpha_values = [0.05]
gamma_values = [0.9]
for alpha in alpha_values:
    for gamma in gamma_values:
        args.append(SarsaZeroForTetris(TetrisEnv(type='fourer', render=False), alpha, gamma))


# trained simultaneously with concurrent.futures
def main(model: SarsaZeroForTetris) -> str:
    """

    :param func_arg: the model to be trained
    :return: string for reporting
    """

    # all with the same learning rate of epsilon
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
                   10, 5, 10)

    return f"Model {model.env.type} gamma:{model.gamma} alpha:{model.alpha} done at {datetime.datetime.now()}"


for arg in args:
    result = main(arg)
    print(result)

# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.submit(main, func_arg) for func_arg in args]
#         for fs in concurrent.futures.as_completed(results):
#             print(fs.result())
