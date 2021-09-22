import datetime
import os
import sys


sys.path.append("/data/leuven/343/vsc34339/RLP")

from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
from Evaluation.train_and_test import train_and_test


path_to_model = "/scratch/leuven/343/vsc34339/RLData/alpha_0.05_gamma_0.9/Model/model.pickle"
path_to_scratch_dir = "/scratch/leuven/343/vsc34339/RLData"
path_to_data_dir = "/data/leuven/343/vsc34339/RLData"
#
path_to_model = r"D:\Bibliotheken\Downloads\RLData\alpha_0.05_gamma_0.9\Model\model.pickle"
path_to_scratch_dir = r"D:\Bibliotheken\Downloads\RLData"
path_to_data_dir = r"D:\Bibliotheken\OneDrive\Documenten\RLData"

model = SarsaZeroAfterStates.load(path_to_model)

name_path = os.path.join(path_to_scratch_dir, f"alpha_{model.alpha}_gamma_{model.gamma}")
data_path = os.path.join(name_path, "Data", "2nd_round")
os.mkdir(data_path)
model_path = os.path.join(name_path, "Model", "2nd_model.pickle")


def epsilon(x: int) -> float:
    return 0.001


train_and_test(model,
               epsilon,
               model_path,
               data_path,
               110, 5, 10)

print(f"Model gamma:{model.gamma} alpha:{model.alpha} done at {datetime.datetime.now()}")
