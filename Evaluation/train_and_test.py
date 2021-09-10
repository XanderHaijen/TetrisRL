import pickle
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt

from Models.Model import Model
from Models.SarsaLambdaForTetris import SarsaLambdaForTetris
from Models.SarsaZeroForTetris import SarsaZeroForTetris
from Models.OnPolicyMCForTetris import OnPolicyMCForTetris
from Evaluate_policy import evaluate_policy, plot_with_errors
import os
import time


def train_and_test(model: Model,
                   learning_rate: Callable[[int], float],
                   model_path: str,
                   metrics_dir: str,
                   training_size: int = 1000,
                   nb_training_sessions: int = 20,
                   eval_size: int = 500) -> None:
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


def finetune_alpha_gamma(test_values_alpha: Union[tuple, list],
                         test_values_gamma: Union[tuple, list],
                         learning_rate: Callable[[int], float],
                         plot_dir: str,
                         data_path: str,
                         training_size: int = 1000,
                         nb_training_sessions: int = 10,
                         eval_size: int = 500) -> None:
    """
    This function produces several plots and data sequences to evaluate the optimal values of the step-size parameter
    alpha and the parameter gamma in the update rule of a sarsa(0) model. By default, the function trains and evaluates
    any pair (alpha, gamma) for 10 sessions of 1000 episodes.
    :param test_values_alpha: the different values of alpha to test
    :param test_values_gamma: the different values of gamma to test
    :param learning_rate: the function for epsilon
    :param data_path: the file path where the data will be saved
    :param plot_dir: the directory in which to save all plots
    :param training_size: the length of one training session in episodes
    :param nb_training_sessions: the amount of sessions of length :param training_size
    :param eval_size: the amount of episodes the policy is evaluated each time
    :return: None. All constructed plots are saved
    """

    comparison = {}  # contains key-value pairs {(alpha, gamma): (episodes, scores, nbs_pieces)}
    scores, nbs_pieces, episodes = [], [], []
    for alpha in test_values_alpha:
        for gamma in test_values_gamma:
            episodes_trained = 0
            scores, nbs_pieces, episodes = [], [], []
            model = SarsaZeroForTetris(alpha=alpha, gamma=gamma)
            for i in range(1, nb_training_sessions + 1):
                model.train(learning_rate, training_size, episodes_trained)
                episodes_trained += training_size
                metrics = evaluate_policy(model, model.env, eval_size)
                mean = metrics.mean()
                std_dev = metrics.std()
                scores.append((mean["Score"], std_dev["Score"]))
                nbs_pieces.append((mean["Nb_pieces"], std_dev["Nb_pieces"]))
                episodes.append(episodes_trained)
            comparison.update({(alpha, gamma): (scores, nbs_pieces)})

    # Construct plots.
    # One plot is for one value of gamma.
    # On that plot, all values of alpha are displayed with their standard deviation
    i = 1
    for _, gamma in comparison.keys():
        for alpha, _ in comparison.keys():
            scores, nbs_pieces = comparison.get((alpha, gamma))
            plt.figure(i)
            plt.errorbar(episodes, [mean for mean, _ in scores], [std_dev for _, std_dev in scores],
                         label=f"alpha={alpha}")

            plt.figure(i + 1)
            plt.errorbar(episodes, [mean for mean, _ in nbs_pieces], [std_dev for _, std_dev in nbs_pieces],
                         label=f"alpha={alpha}")

        plt.figure(i)
        plt.title(f"Score for gamma={gamma}")
        plt.legend()
        path = os.path.join(plot_dir, f"plot_{i}.jpg")
        plt.savefig(path)

        plt.figure(i + 1)
        plt.title(f"Number of pieces for gamma={gamma}")
        plt.legend()
        path = os.path.join(plot_dir, f"plot_{i + 1}.jpg")
        plt.savefig(path)

        i += 2

    with open(data_path, 'wb') as f:
        pickle.dump(comparison, f)
        f.close()


def main():
    # part to finetune alpha and gamma
    finetune_alpha_gamma((0.1, 0.05, 0.2), (0.8, 0.9, 0.95), lambda x: 1 / x, "plot", "data")

    # part to train and test MC and Sarsa models
    model = SarsaZeroForTetris(0.1, 0.5)
    train_and_test(model, lambda x: 1 / (1 + x), "path_1", "path_2")

    model = SarsaLambdaForTetris(0.9, 0.1, 0.5, "accumulating")
    train_and_test(model, lambda x: 1 / (1 + x), "path_1", "path_2")

    model = OnPolicyMCForTetris(0.9, first_visit=True)
    train_and_test(model, lambda x: 1 / (1 + x), "path_1", "path_2")

    model = OnPolicyMCForTetris(0.9, first_visit=False)
    train_and_test(model, lambda x: 1 / (1 + x), "path_1", "path_2")
