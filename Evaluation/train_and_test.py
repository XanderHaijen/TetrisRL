import pickle
from typing import Callable

from Models.Model import Model
from Models.SarsaLambdaForTetris import SarsaLambdaForTetris
from Models.SarsaZeroForTetris import SarsaZeroForTetris
from Models.OnPolicyMCForTetris import OnPolicyMCForTetris
from Evaluate_policy import evaluate_policy, plot_with_errors
import os
import time

# This file trains and tests the policies in the same fashion:
# By default, 20 training rounds of 1000 games each, with 20 evaluation rounds of 500 games in between
# It saves all evaluation and training data at the provided paths
# Evaluation data saved:
#   • the number of pieces placed using the policy after each 1000-game interval (mean and standard deviation)
#   • the score obtained using the policy after each 1000-game interval (mean and standard deviation)
# Training data saved:
#   • the time it took the algorithm to achieve an average score of 200 and 1000 (in seconds).
#       If this score was not reached after the whole of training, the time is of type str, containing "Inf"


def train_and_test(model: Model,
                   learning_rate: Callable[[int], float],
                   model_path: str,
                   metrics_dir: str,
                   training_size: int = 1000,
                   nb_training_sessions: int = 20,
                   eval_size: int = 500):
    episodes_trained = 0
    scores, nbs_pieces, episodes = [], [], []
    reached_200, reached_1000 = False, False
    t1, t2 = None, None

    t0 = time.time()
    for i in range(1, nb_training_sessions + 1):
        model.train(learning_rate, training_size, episodes_trained)
        episodes_trained += training_size
        metrics = evaluate_policy(model, model.env, eval_size)
        mean = metrics.mean()
        if not reached_200 and mean["Score"] > 200:
            t1 = time.time()
            reached_200 = True
        if not reached_1000 and mean["Score"] > 1000:
            t2 = time.time()
            reached_1000 = True
        std_dev = metrics.std()
        scores.append((mean["Score"], std_dev["Score"]))
        nbs_pieces.append((mean["Nb_pieces"], std_dev["Nb_pieces"]))
        episodes.append(episodes_trained)

    # save the measured times
    if t1 is not None:
        time_to_reach_200 = t1 - t0
    else:
        time_to_reach_200 = "Inf"

    if t2 is not None:
        time_to_reach_1000 = t2 - t0
    else:
        time_to_reach_1000 = "Inf"

    times = [time_to_reach_200, time_to_reach_1000]

    path = os.path.join(metrics_dir, "times.pickle")
    with open(path, 'wb') as f:
        pickle.dump(times, f)
        f.close()

    # save the model
    model.save(model_path)

    # save the evaluation data
    path = os.path.join(metrics_dir, "scores_and_pieces.pickle")
    with open(path, 'wb') as f:
        pickle.dump((episodes, scores, nbs_pieces), f)
        f.close()

    # plot the figure for the score
    path = os.path.join(metrics_dir, "score_plot.jpg")
    plot_with_errors(episodes, scores, path)

    # plot the figure for the nbs_pieces
    path = os.path.join(metrics_dir, "score_plot.jpg")
    plot_with_errors(episodes, nbs_pieces, path)


model = SarsaZeroForTetris(0.1, 0.5)
train_and_test(model, lambda x: 1 / (1+x), "path_1", "path_2")

model = SarsaLambdaForTetris(0.9, 0.1, 0.5, "accumulating")
train_and_test(model, lambda x: 1 / (1+x), "path_1", "path_2")

model = OnPolicyMCForTetris(0.9, first_visit=True)
train_and_test(model, lambda x: 1 / (1+x), "path_1", "path_2")

model = OnPolicyMCForTetris(0.9, first_visit=False)
train_and_test(model, lambda x: 1 / (1+x), "path_1", "path_2")
