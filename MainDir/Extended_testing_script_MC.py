import sys

sys.path.append("/data/leuven/343/vsc34339/TetrisRL-master")

from Evaluation.Extended_test import *
from Models.OnPolicyMCForTetris import OnPolicyMCForTetris

models_dir = "/scratch/leuven/343/vsc34339/MC_fourer_SV"
target_dir = "/data/leuven/343/vsc34339/RLData/MC_fourer_SV_results"
extended_test_state_action(OnPolicyMCForTetris, models_dir, target_dir, 10000)
