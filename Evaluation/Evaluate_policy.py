import time

import pandas as pd
import numpy as np

from Models.Model import Model
from gym import Env


def evaluate_policy(model: Model, env: Env, nb_episodes: int, render: bool = False) -> pd.DataFrame():
    """

    :param render: if True, the games will be shown on screen
    :param model: of type Model: provides the policy to follow
    :param env: the environment in which to test the provided :param model
    :param nb_episodes: number of evaluation epsiodes
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
            if render:  # only for viewing purposes
                env.render()
                time.sleep(0.05)
            actions = model.predict(state)
            for action in actions:
                state, reward, done, data = env.step(action)
                if data["new_piece"]:
                    nb_pieces += 1
                total_cleared += data["lines_cleared"]

        lines_cleared = data.get("lines_cleared", 0)
        score = data.get("score", 0)
        metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": lines_cleared, "Score": score})

    metrics_df = pd.DataFrame.from_records(metrics)
    return metrics_df




