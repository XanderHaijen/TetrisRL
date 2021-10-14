import concurrent.futures
import datetime
import os
import shutil
import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.SarsaLambdaAfterstates import SarsaLambdaAfterstates
from tetris_environment.tetris_env import TetrisEnv

args = []

path_to_data_dir = "/scratch/leuven/343/vsc34339/RLData/SarsaLambda"
# path_to_data_dir = r'C:\Users\xande\Downloads'


# This file will train and test several combinations of alpha, gamma and lambda
alpha_values = [0.05]
gamma_values = [0.9]
lambda_values = [0]

# With a given type of traces
traces_values = ["accumulating", "dutch", "replacing"]

# On a predefined board size
size = "fourer"
env = TetrisEnv(type=size)


for alpha in alpha_values:
    for gamma in gamma_values:
        for Lambda in lambda_values:
            for traces in traces_values:
                args.append(SarsaLambdaAfterstates(env, Lambda, alpha, gamma, traces))


# trained simultaneously with concurrent.futures
def main(func_arg: SarsaLambdaAfterstates) -> str:
    """

    :param func_arg: the model to be trained
    :return: string for reporting
    """

    # all with the same learning rate of epsilon
    def epsilon(x: int) -> float:
        return 0.001

    # Unpack func_args
    model: SarsaLambdaAfterstates = func_arg

    name_path = os.path.join(path_to_data_dir, f"{model.env.type}_alpha_{model.alpha}_gamma_{model.gamma}_"
                                               f"lambda_{model.Lambda}_with_{model.traces}_traces")
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

    return f"{model.env.type}_alpha_{model.alpha}_gamma_{model.gamma}_lambda_{model.Lambda}_with_{model.traces}_traces" \
           f" done at {datetime.datetime.now()}."


for arg in args:
    result = main(arg)
    print(result)

# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = [executor.submit(main, func_arg) for func_arg in args]
#         for fs in concurrent.futures.as_completed(results):
#             print(fs.result())
