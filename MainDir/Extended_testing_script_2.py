import sys

sys.path.append("/data/leuven/343/vsc34339/TetrisRL-master")

from Evaluation.Extended_test import *
from Models.SarsaLambdaForTetris import SarsaLambdaForTetris

models_dir = "/scratch/leuven/343/vsc34339/SarsaLambda_SV"
target_dir = "/data/leuven/343/vsc34339/RLData/SarsaLambda_fourer_SV_results"
extended_test_state_action(SarsaLambdaForTetris, models_dir, target_dir, 10000)
