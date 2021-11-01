import pickle
from typing import Callable, Union
from Evaluation import Evaluate_policy
from Models.AfterstateModel import AfterstateModel
from Models.StateValueModel import StateValueModel
import os
import datetime
import time
import pandas as pd


def train_and_test(model: Union[StateValueModel, AfterstateModel],
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
    :return: None
    All computed values, including the returns, are saved to file.
    """
    print(f"starting train and test for {model} at {datetime.datetime.now()}")
    all_metrics = [["Episodes trained", "Lines mean", "Lines lower quantile", "Lines upper quantile", "Nb pieces mean",
                    "Nb pieces lower quantile", "Nb pieces upper quantile", "Score mean",
                    "Score lower quantile", "Score upper quantile"]]
    episodes_trained = ep_trained
    reached_200, reached_1000 = False, False
    t1, t2 = None, None

    t0 = time.perf_counter()
    for i in range(1, nb_training_sessions + 1):
        # print(f"starting training round {i} at {datetime.datetime.now()}")
        model.train(learning_rate, training_size, episodes_trained)
        episodes_trained += training_size

        # print(f"starting evaluations round {i} at {datetime.datetime.now()}")
        if isinstance(model, AfterstateModel):
            metrics = Evaluate_policy.evaluate_policy_afterstates(model, model.env, eval_size)
        else:  # if isinstance(model, StateValueModel):
            metrics = Evaluate_policy.evaluate_policy_state_action(model, model.env, eval_size)
        mean = metrics.mean()
        if not reached_200 and mean["Lines_cleared"] > 20:
            t1 = time.perf_counter()
            reached_200 = True
        if not reached_1000 and mean["Lines_cleared"] > 100:
            t2 = time.perf_counter()
            reached_1000 = True
        quantiles = metrics.quantile([0.25, 0.75])
        all_metrics.append([episodes_trained,
                            mean["Lines_cleared"], quantiles["Lines_cleared"][0.25], quantiles["Lines_cleared"][0.75],
                            mean["Nb_pieces"], quantiles["Nb_pieces"][0.25], quantiles["Nb_pieces"][0.75],
                            mean["Score"], quantiles["Score"][0.25], quantiles["Score"][0.75]])
        model.save(model_path)
        # print(f"ending round {i} at {datetime.datetime.now()}. Average score is {mean['Score']}.")
    # print(f"starting afterprocessing at {datetime.datetime.now()}")

    dataframe = pd.DataFrame(all_metrics[1:], columns=all_metrics[0])

    # save the measured times
    if t1 is not None:
        time_to_clear_20 = t1 - t0
    else:
        time_to_clear_20 = "Inf"

    if t2 is not None:
        time_to_clear_100 = t2 - t0
    else:
        time_to_clear_100 = "Inf"

    path = os.path.join(metrics_dir, "times.txt")
    with open(path, 'w+') as f:
        f.write(f"Time to reach 200: {time_to_clear_20} \n")
        f.write(f"Time to reach 1,000: {time_to_clear_100} \n")
        f.close()

    # save the model
    model.save(model_path)

    # save the evaluation data
    path = os.path.join(metrics_dir, "scores_and_pieces.csv")
    dataframe.to_csv(path)

    # print("plotting figures")
    # plot the figure for the score
    j = 1
    path = os.path.join(metrics_dir, "lines_plot.jpg")
    top_err = list(dataframe["Lines upper quantile"] - dataframe["Lines mean"])
    low_err = list(dataframe["Lines mean"] - dataframe["Lines lower quantile"])
    print(top_err)
    Evaluate_policy.plot_with_errors(dataframe["Episodes trained"], dataframe["Lines mean"],
                                     (low_err, top_err), "average lines cleared", path, j)

    # plot the figure for the nbs_pieces
    j += 1
    path = os.path.join(metrics_dir, "pieces_plot.jpg")
    top_err = dataframe["Nb pieces upper quantile"] - dataframe["Nb pieces mean"]
    low_err = dataframe["Nb pieces mean"] - dataframe["Nb pieces lower quantile"]
    Evaluate_policy.plot_with_errors(dataframe["Episodes trained"], dataframe["Nb pieces mean"], (top_err, low_err),
                                     "average number of pieces", path, j)
