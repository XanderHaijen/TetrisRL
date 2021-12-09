import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir = r"C:\Users\xande\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData\SarsaLambda_exfourer_AS_LearningCurves"
for file in os.listdir(dir):
    path = os.path.join(dir, file, "Data", "scores_and_pieces.csv")
    print(path)
    data = pd.read_csv(path)
    ep_trained = list(data["Episodes trained"])
    ep_trained.insert(0,0)
    nb_lines = list(data["Lines mean"])
    nb_lines.insert(0, 1.94)
    label = "accumulating, lambda= " if file.__contains__("accumulating") else \
            "dutch, lambda= " if file.__contains__("dutch") else \
            "replacing, lambda= "
    label += ('0.95' if file.__contains__("0.95") else '1')
    new_lines = [0 for _ in range(len(nb_lines))]
    new_lines[0] = nb_lines[0]
    for i in range(1, len(ep_trained) - 1):
        new_lines[i] = (nb_lines[i - 1] + nb_lines[i] + nb_lines[i + 1]) / 3
    new_lines[-1] = (nb_lines[-2] + nb_lines[-1]) / 2
    plt.figure(1)
    plt.plot(ep_trained, nb_lines, marker='o', label=label)
    plt.figure(2)
    plt.plot(ep_trained, new_lines, marker='o', label=label)

plt.figure(1)
plt.legend()
plt.ylabel("Number of lines cleared")
plt.xlabel("Number of episodes trained (x10,000)")
plt.title("Some learning curves for Sarsa Lambda agents in extended fourer")
x_ticks = np.linspace(start=0, stop=20, num=11, dtype=int)
x_pos = np.linspace(start=0, stop=200000, num=11)
plt.xticks(x_pos, x_ticks)
plt.grid(axis='y', linestyle='--', lw=0.5)
plt.xlim(left=0, right=200000)
plt.figure(2)
plt.legend()
plt.ylabel("Averaged number of lines cleared")
plt.xlabel("Number of episodes trained (x10,000)")
plt.title("Smoothened learning curves for Sarsa Lambda agents in extended fourer")
x_ticks = np.linspace(start=0, stop=20, num=11, dtype=int)
x_pos = np.linspace(start=0, stop=200000, num=11)
plt.xticks(x_pos, x_ticks)
plt.grid(axis='y', linestyle='--', lw=0.5)
plt.xlim(left=0, right=200000)

plt.show()
