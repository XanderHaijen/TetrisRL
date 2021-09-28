import time
import matplotlib.pyplot as plt
import pandas as pd
from Models.StateValueModel import StateValueModel
from gym import Env


def evaluate_policy_afterstates(algorithm: StateValueModel, env: Env, nb_episodes: int) -> pd.DataFrame():
    """

    :param algorithm: of type StateValueModel: provides the policy to follow
    :param env: the environment in which to test the provided :param algorithm
    :param nb_episodes: number of evaluation episodes
    :return: returns the mean and variance of number of pieces placed, number of lines cleared and score achieved
    """

    # metrics will be constructed as a list of dicts, then converted to pandas dataframe for analysis
    # one dict will contain nb_pieces, score and lines_cleared

    metrics = []
    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        done = False
        nb_pieces = 0
        j = 0
        data = {}
        total_cleared = 0
        while not done:
            j += 1
            actions = algorithm.predict(state)
            for action in actions:
                state, reward, done, data = env.step(action)
                total_cleared += data["lines_cleared"]
            nb_pieces += 1

        score = data.get("score", 0)
        metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": total_cleared, "Score": score})

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
    # one dict will contain nb_pieces, score and lines_cleared

    metrics = []
    for episode in range(1, nb_episodes + 1):
        state = env.reset()
        done = False
        nb_pieces = 0
        j = 0
        data = {}
        total_cleared = 0
        while not done:
            j += 1
            action = algorithm.predict(state)
            state, reward, done, data = env.step(action)
            if data["new_piece"]:
                nb_pieces += 1
            total_cleared += data["lines_cleared"]

        score = data.get("score", 0)
        metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": total_cleared, "Score": score})

    metrics_df = pd.DataFrame.from_records(metrics)
    return metrics_df


def plot_with_errors(x_sequence, y_sequence, name, image_path, fig_nb) -> None:
    """
    :param name: name to put on the y-axis and in the label of the legend
    :param fig_nb: ensures plots are not overwritten
    :param x_sequence: list containing moments of measure (in number of episodes)
    :param y_sequence: list or tuple containing a number of pairs (mean, variance)
    :param image_path: path where the figure is saved
    :return: None
    """
    plt.figure(fig_nb)
    plt.errorbar(x_sequence, [mean for mean, _ in y_sequence], [var for _, var in y_sequence],
                 linestyle='-', marker='^', label=name)
    plt.legend()
    plt.xlabel("Number of episodes trained")
    plt.ylabel(name)
    plt.savefig(image_path)
