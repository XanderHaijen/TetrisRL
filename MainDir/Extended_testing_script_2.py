import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.Extended_test import extended_test
from Models.OnPolicyMCAfterstates import OnPolicyMCAfterstates

models_dir = "/data/leuven/343/vsc34339/MonteCarloExFourer"
target_dir = "/data/leuven/343/vsc34339/RLData/MC_exfourer_AS_results"
extended_test(OnPolicyMCAfterstates, models_dir, target_dir, 10000)
