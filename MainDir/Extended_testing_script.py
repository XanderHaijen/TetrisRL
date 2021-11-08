import sys

sys.path.append("/data/leuven/343/vsc34339/TetrisRL-master")

from Evaluation.Extended_test import extended_test
from Models.SarsaLambdaAfterstates import SarsaLambdaAfterstates

models_dir = "/scratch/leuven/343/vsc34339/SarsaLambda"
target_dir = "/data/leuven/343/vsc34339/RLData/SarsaLambda_fourer_AS_results"
extended_test(SarsaLambdaAfterstates, models_dir, target_dir, 10000)
