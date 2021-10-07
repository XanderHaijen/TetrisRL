import datetime
import os
import sys


sys.path.append("/data/leuven/343/vsc34339/RLP")

from Models.OnPolicyMCAfterstates import OnPolicyMCAfterstates
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
from Evaluation.train_and_test import train_and_test


path_to_model = "/scratch/leuven/343/vsc34339/RLData/MonteCarlo/fourer_MC_gamma_0.9/Model/model.pickle"
path_to_scratch_dir = "/scratch/leuven/343/vsc34339/RLData"
path_to_data_dir = "/data/leuven/343/vsc34339/RLData"
#

model = OnPolicyMCAfterstates.load(path_to_model)

name_path = os.path.join(path_to_scratch_dir, "MonteCarlo", "fourer_MC_gamma_0.9")
data_path = os.path.join(name_path, "2nd_round", "Data")
os.mkdir(data_path)
model_path = os.path.join(name_path, "2nd_round", "2nd_model.pickle")


def epsilon(x: int) -> float:
    return 0.001


train_and_test(model,
               epsilon,
               model_path,
               data_path,
               10000, 17, 1000)

print(f"{model} done at {datetime.datetime.now()}")
