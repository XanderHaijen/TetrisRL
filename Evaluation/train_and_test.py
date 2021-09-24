import pickle
from typing import Callable
from Evaluation import Evaluate_policy
from Models.Model import Model
import os
import datetime
import time


def train_and_test(model: Model,
                   learning_rate: Callable[[int], float],
                   model_path: str,
                   metrics_dir: str,
                   training_size: int = 1000,
                   nb_training_sessions: int = 20,
                   eval_size: int = 500,
                   ep_trained: int = 0) -> None:
    """
    This file trains and tests the policies in the same fashion:
    By default, 20 training rounds of 1000 games each, with 20 evaluation rounds of 500 games in between
    It saves all evaluation and training data at the provided paths
    Evaluation data saved:
        • the number of pieces placed using the policy after each 1000-game interval (mean and standard deviation)
        • the score obtained using the policy after each 1000-game interval (mean and standard deviation)
    Training data saved:
        • the time it took the algorithm to achieve an average score of 200 and 1000 (in seconds).
            If this score was not reached after the whole of training, the time is of type str, containing "Inf"
    :param ep_trained: for training an already (partially) trained agent
    :param model: the model to train and test
    :param learning_rate: the function for epsilon
    :param model_path: the file path where the model will be saved
    :param metrics_dir: the directory in which to save all measurements
    :param training_size: the length of one training session in episodes
    :param nb_training_sessions: the amount of sessions of length :param training_size
    :param eval_size: the amount of episodes the policy is evaluated each time
    :return: the evaluation data for nbs_pieces and the score.
    All computed values, including the returns, are saved to file.
    """
    print("starting train and test")
    episodes_trained = ep_trained
    scores, nbs_pieces, episodes = [], [], []
    reached_200, reached_1000 = False, False
    t1, t2 = None, None

    t0 = time.perf_counter()
    for i in range(1, nb_training_sessions + 1):
        print(f"starting training round {i} at {datetime.datetime.now()}")
        model.train(learning_rate, training_size, episodes_trained)
        episodes_trained += training_size
        print(f"starting evaluations round {i} at {datetime.datetime.now()}")
        metrics = Evaluate_policy.evaluate_policy_afterstates(model, model.env, eval_size)
        mean = metrics.mean()
        if not reached_200 and mean["Score"] > 200:
            t1 = time.perf_counter()
            reached_200 = True
        if not reached_1000 and mean["Score"] > 1000:
            t2 = time.perf_counter()
            reached_1000 = True
        std_dev = metrics.std()
        scores.append((mean["Score"], std_dev["Score"]))
        nbs_pieces.append((mean["Nb_pieces"], std_dev["Nb_pieces"]))
        episodes.append(episodes_trained)
        model.save(model_path)
        print(f"ending round {i} at {datetime.datetime.now()}. Average score is {mean['Score']}.")

    print(f"starting afterprocessing at {datetime.datetime.now()}")

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

    print("plotting figures")
    # plot the figure for the score
    j = 1
    path = os.path.join(metrics_dir, "score_plot.jpg")
    Evaluate_policy.plot_with_errors(episodes, scores, "average score", path, j)

    # plot the figure for the nbs_pieces
    j += 1
    path = os.path.join(metrics_dir, "pieces_plot.jpg")
    Evaluate_policy.plot_with_errors(episodes, nbs_pieces, "average number of pieces", path, j)
