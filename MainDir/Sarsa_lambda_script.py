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

path_to_data_dir = "/scratch/leuven/343/vsc34339/RLData/SarsaLambda/"

# On a predefined board size
size = "extended fourer"
env = TetrisEnv(type=size)

args.append(SarsaLambdaAfterstates(env, 0.8, 0.1, 0.7, "replacing"))
args.append(SarsaLambdaAfterstates(env, 0.9, 0.1, 0.7, "replacing"))


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
                   1, 2, 1)

    return f"{model.env.type}_alpha_{model.alpha}_gamma_{model.gamma}_lambda_{model.Lambda}_with_{model.traces}_traces" \
           f" done at {datetime.datetime.now()}."


# for arg in args:
#     result = main(arg)
#     print(result)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(main, args):
            print(result)
