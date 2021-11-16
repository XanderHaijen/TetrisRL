import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir = r"C:\Users\xande\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData\SarsaLambda_AS_fourer_LearningCurves"
for file in os.listdir(dir):
    path = os.path.join(dir, file, "Data", "scores_and_pieces.csv")
    print(path)
    data = pd.read_csv(path)
    ep_trained = list(data["Episodes trained"])
    ep_trained.insert(0,0)
    nb_pieces = list(data["Nb pieces mean"])
    nb_pieces.insert(0, 26.15)
    label = "accumulating, lambda= " if file.__contains__("accumulating") else \
            "dutch, lambda= " if file.__contains__("dutch") else \
            "replacing, lambda= "
    label += ('0.8' if file.__contains__("0.8") else '1')
    new_pieces = [0 for _ in range(len(nb_pieces))]
    new_pieces[0] = nb_pieces[0]
    for i in range(1, len(ep_trained) - 1):
        new_pieces[i] = (nb_pieces[i-1] + nb_pieces[i] + nb_pieces[i+1]) / 3
    new_pieces[-1] = (nb_pieces[-2] + nb_pieces[-1]) / 2

    plt.plot(ep_trained, new_pieces, marker='x', label=label)

plt.legend()
plt.ylabel("Number of pieces placed")
plt.xlabel("Number of episodes trained (x10,000)")
plt.title("Some learning curves for Sarsa Lambda agents in fourer")
x_ticks = np.linspace(start=0, stop=20, num=11, dtype=int)
x_pos = np.linspace(start=0, stop=200000, num=11)
plt.xticks(x_pos, x_ticks)
plt.grid(axis='y', linestyle='--', lw=0.5)
plt.xlim(left=0, right=200000)
plt.show()
