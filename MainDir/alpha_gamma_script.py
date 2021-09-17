import concurrent.futures
import os
import shutil
import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import train_and_test
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates

args = []

path_to_scratch_dir = "/scratch/leuven/343/vsc34339/RLData"
path_to_data_dir = "/data/leuven/343/vsc34339/RLData"
#
# path_to_scratch_dir = r"D:\Bibliotheken\Downloads\RLData"
# path_to_data_dir = r"D:\Bibliotheken\OneDrive\Documenten\RLData"

# This file will train and test several combinations of alpha and gamma using a Sarsa(0) model
alpha_values = [0.05]
gamma_values = [0.9]
for alpha in alpha_values:
    for gamma in gamma_values:
        args.append(SarsaZeroAfterStates(alpha, gamma))


# trained simultaneously with concurrent.futures
def main(func_arg: SarsaZeroAfterStates) -> str:
    """

    :param func_arg: the model to be trained
    :return: string for reporting
    """

    # all with the same learning rate of epsilon
    def epsilon(x):
        return 1 / (1 + x)

    # Unpack func_args
    model: SarsaZeroAfterStates = func_arg

    name_path = os.path.join(path_to_scratch_dir, f"alpha_{model.alpha}_gamma_{model.gamma}")
    data_path = os.path.join(name_path, "Data")
    model_dir = os.path.join(name_path, "Model")

    os.mkdir(name_path)
    os.mkdir(data_path)
    os.mkdir(model_dir)

    model_path = os.path.join(model_dir, "model.pickle")

    train_and_test(model,
                   epsilon,
                   model_path,
                   data_path,
                   10, 5, 10)

    return f"Model gamma:{model.gamma} alpha:{model.alpha} done"


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(main, func_arg) for func_arg in args]
        for fs in concurrent.futures.as_completed(results):
            print(fs.result())

    shutil.copy(path_to_scratch_dir, path_to_data_dir)
