import time

import pandas as pd
import numpy as np

from Algorithms.Algorithm import Algorithm
from gym import Env


def evaluate_policy(algorithm: Algorithm, env: Env, nb_episodes: int):
    """

    :param algorithm: of type Algorithm: provides the policy to follow
    :param env: the environment in which to test the provided :param algorithm
    :param nb_episodes: number of evaluation epsiodes
    :return: returns the mean and variance of number of pieces placed, number of lines cleared and score achieved
    """

    # metrics will be constructed as a list of dicts, then converted to pandas dataframe for analysis
    # one dict will contain nb_pieces, score and lines_cleared

    metrics = []
    for episode in range(1, nb_episodes + 1):
        observations = env.reset()
        done = False
        nb_pieces = 0
        j = 0
        data = {}
        while not done:
            j += 1
            # env.render()  # only for viewing purposes
            action = algorithm.predict()
            state, reward, done, data = env.step(action)
            if data["new_piece"]:
                nb_pieces += 1

        lines_cleared = data.get("lines_cleared", 0)
        score = data.get("score", 0)
        metrics.append({"Nb_pieces": nb_pieces, "Lines_cleared": lines_cleared, "Score": score})

    metrics_df = pd.DataFrame.from_records(metrics)


