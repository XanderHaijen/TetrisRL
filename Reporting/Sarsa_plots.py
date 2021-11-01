# Given a directory with CSVs
import os

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import csv

import numpy as np
import pandas as pd

source_dir = r"C:\Users\xande\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData\Sarsa0_exfourer_AS_results\Results"


def is_csv(filepath) -> bool:
    try:
        with open(filepath, newline='') as csvfile:
            csv.reader(csvfile, delimiter=' ', quotechar='|')
    except:
        return False
    else:
        return True


def find_model_type(filename):
    start_alpha = filename.find("alpha=")
    i = 6
    alpha = ""
    while filename[start_alpha + i] == "." or str.isdigit(filename[start_alpha + i]):
        alpha += filename[start_alpha + i]
        i += 1

    start_gamma = filename.find("gamma=")
    i = 6
    gamma = ""
    while filename[start_gamma + i] == "." or str.isdigit(filename[start_gamma + i]):
        gamma += filename[start_gamma + i]
        i += 1

    return float(alpha), float(gamma)


def quantile_plot(stats: dict, xlabel, ylabel):
    """
    DOES NOT SAVE THE PLOT OR RESET IT
    """
    gamma_values = {key[1] for key in stats.keys()}
    nb_gamma = len(gamma_values)
    i = 1
    plt.title(f"Quantile plot for {ylabel.lower()}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for gamma in gamma_values:
        models_subset = [(key, value) for key, value in stats.items() if key[1] == gamma]
        models_subset.sort(key=lambda key: key[0][0])
        x_axis = list(range(len(models_subset)))
        x_axis = [x + displacement(i, nb_gamma) for x in x_axis]
        means = [value[1] for _, value in models_subset]
        lower = [value[0] for _, value in models_subset]
        upper = [value[2] for _, value in models_subset]
        alpha_values = [key[0] for key, _ in models_subset]
        plt.errorbar(x=x_axis, y=means, yerr=(lower, upper), fmt='x', label=f"gamma={gamma}")
        i += 1
    plt.xticks(range(len(alpha_values)), alpha_values)
    plt.grid(axis='y', linestyle='--', lw=0.5)
    plt.ylim(bottom=0)
    plt.legend()


def displacement(i, numbers, width=0.25):
    """
    Gives the offset needed for the i-th element to fit in a -0.2,0.2 range :param numbers elements
    :param width:
    :param i:
    :param numbers:
    :return:
    """
    return np.linspace(start=-width, stop=width, num=numbers)[i-1]

def main(source_dir):
    lines_stats = {}
    pieces_stats = {}
    params = []
    (_, _, files) = os.walk(source_dir).__next__()
    for file in files:
        print(file)
        path = os.path.join(source_dir, file)
        if is_csv(path):
            print(True)
            try:
                alpha, gamma = find_model_type(file)
            except ValueError:
                continue

            if True:
                data = pd.read_csv(path)
                mean = data.mean()
                quantiles = data.quantile([0.25, 0.75])
                pieces_stats.update({(alpha, gamma): (mean["Nb_pieces"] - quantiles["Nb_pieces"][0.25],
                                                      mean["Nb_pieces"],
                                                      quantiles["Nb_pieces"][0.75] - mean["Nb_pieces"])})
                lines_stats.update({(alpha, gamma): (mean["Lines_cleared"] - quantiles["Lines_cleared"][0.25],
                                                     mean["Lines_cleared"],
                                                     quantiles["Lines_cleared"][0.75] - mean["Lines_cleared"])})

    plt.figure(1)
    quantile_plot(lines_stats, "Alpha", "Lines cleared")
    plt.figure(2)
    quantile_plot(pieces_stats, "Alpha", "Pieces placed")
    plt.show()


main(source_dir)
