import time
import matplotlib.pyplot as plt
import pandas as pd

from Models.AfterstateModel import AfterstateModel
from Models.StateActionModel import StateValueModel
from gym import Env


def evaluate_policy_afterstates(algorithm: AfterstateModel, env: Env, nb_episodes: int) -> pd.DataFrame():
    """

    :param algorithm: of type StateValueModel: provides the policy to follow
    :param env: the environment in which to test the provided :param algorithm
    :param nb_episodes: number of evaluation episodes
    :return: returns the mean and variance of number of pieces placed, number of lines cleared and score achieved
    """

    # metrics will be constructed as a list of dicts, then converted to pandas dataframe for analysis
    # one dict will contain nb_lines, score and lines_cleared

    metrics = []
    for episode in range(1, nb_episodes + 1):
        total_score = 0
        total_cleared = 0
        nb_pieces = 0
        state = env.reset()
        done = False
        while not done:
            _, actions = algorithm.predict()
            for action in actions:
                state, reward, done, data = env.step(action)
                if done:
                    break
                total_cleared += data["lines_cleared"]
                total_score += data["score"]
            nb_pieces += 1

        metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": total_cleared, "Score": total_score})

    metrics_df = pd.DataFrame.from_records(metrics)
    return metrics_df


def evaluate_policy_state_action(algorithm: StateValueModel, env: Env, nb_episodes: int) -> pd.DataFrame():
    """
    :param algorithm: of type StateValueModel: provides the policy to follow
    :param env: the environment in which to test the provided :param algorithm
    :param nb_episodes: number of evaluation episodes
    :return: returns the mean and variance of number of pieces placed, number of lines cleared and score achieved
    """

    # metrics will be constructed as a list of dicts, then converted to pandas dataframe for analysis
    # one dict will contain nb_lines, score and lines_cleared

    metrics = []
    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        done = False
        nb_pieces = 0
        j = 0
        data = {}
        total_cleared = 0
        total_score = 0
        while not done:
            j += 1
            action = algorithm.predict(state)
            state, reward, done, data = env.step(action)
            if data["new_piece"]:
                nb_pieces += 1
            total_cleared += data["lines_cleared"]
            total_score += data["score"]

        metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": total_cleared, "Score": total_score})

    metrics_df = pd.DataFrame.from_records(metrics)
    return metrics_df


def plot_with_errors(x_sequence, y_sequence, errors, name, image_path, fig_nb) -> None:
    """
    :param name: name to put on the y-axis and in the label of the legend
    :param fig_nb: ensures plots are not overwritten
    :param x_sequence: list containing moments of measure (in number of episodes)
    :param y_sequence: list or tuple containing a number of pairs (mean, variance)
    :param image_path: path where the figure is saved
    :return: None
    """
    plt.figure(fig_nb)
    plt.errorbar(x_sequence, y_sequence, yerr=errors,
                 linestyle='-', marker='x', label=name)
    plt.legend()
    plt.xlabel("Number of episodes trained")
    plt.ylabel(name)
    plt.savefig(image_path)
