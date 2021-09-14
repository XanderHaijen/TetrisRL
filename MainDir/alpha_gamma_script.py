import os
import shutil
import sys

sys.path.append("/data/leuven/343/vsc34339/RLP")

from Evaluation.train_and_test import finetune_alpha_gamma

path_to_scratch_dir = "/scratch/leuven/343/vsc34339"
path_to_data_dir = "/data/leuven/343/vsc34339"

plots_paths = os.path.join(path_to_scratch_dir, "Plots")
data_path = os.path.join(path_to_scratch_dir, "Data", "eval_data.pickle")
model_dir = os.path.join(path_to_scratch_dir, "Models")

finetune_alpha_gamma([0.1, 0.05],
                     [0.9, 0.8],
                     lambda x: 1/(1+x),
                     plots_paths,
                     data_path,
                     model_dir,
                     1000, 10, 750)

# move plots and data to data directory
shutil.copy(plots_paths, os.path.join(path_to_data_dir, "Plots"))
shutil.copy(data_path, os.path.join(path_to_data_dir, "Data"))
shutil.copy(model_dir, os.path.join(path_to_data_dir, "Models"))
