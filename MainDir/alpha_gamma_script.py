import os
import shutil

from Evaluation.train_and_test import finetune_alpha_gamma
path_to_scratch_dir = "$VSC_SCRATCH"
path_to_data_dir = "$VSC_DATA"

plots_paths = os.path.join(path_to_scratch_dir, "Plots")
data_path = os.path.join(path_to_scratch_dir, "data", "eval_data.pickle")

finetune_alpha_gamma([0.1, 0.05],
                     [0.9, 0.8],
                     lambda x: 1/(1+x),
                     plots_paths,
                     data_path,
                     1000, 10, 750)

# move plots and data to data directory
shutil.move(plots_paths, path_to_data_dir)
shutil.move(data_path, os.path.join(path_to_data_dir, "data"))
