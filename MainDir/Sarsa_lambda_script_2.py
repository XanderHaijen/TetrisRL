import concurrent.futures
import datetime
import os
import shutil
import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.SarsaLambdaForTetris import SarsaLambdaForTetris
from tetris_environment.tetris_env import TetrisEnv

args = []

path_to_data_dir = "/scratch/leuven/343/vsc34339/RLData/SarsaLambda_AS_exfourer"
# path_to_data_dir = r"D:\Bibliotheken\Downloads"


# This upper_dir will train and test several combinations of alpha, gamma and lambda
alpha_values = [0.1]
gamma_values = [0.7]
lambda_values = [1, 0.95, 0.9, 0.8]

# With a given type of traces
traces_values = ["accumulating"]

# On a predefined board size
size = "extended fourer"
env = TetrisEnv(type=size)


for alpha in alpha_values:
    for gamma in gamma_values:
        for Lambda in lambda_values:
            for traces in traces_values:
                args.append(SarsaLambdaForTetris(env, Lambda, alpha, gamma, traces))


# trained simultaneously with concurrent.futures
def main(model: SarsaLambdaForTetris) -> str:
    """

    :param model: the model to be trained
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
                   1, 2, 1)

    return f"{model} done at {datetime.datetime.now()}."


# for arg in args:
#     result = main(arg)
#     print(result)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(main, args):
            print(result)
