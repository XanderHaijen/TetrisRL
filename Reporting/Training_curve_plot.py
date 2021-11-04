import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir = r"C:\Users\xande\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData\MC_fourer_SV_learningCurves"
for file in os.listdir(dir):
    path = os.path.join(dir, file, "Data", "scores_and_pieces.csv")
    data = pd.read_csv(path)
    ep_trained = list(data["Episodes trained"])
    ep_trained.insert(0,0)
    nb_pieces = list(data["Nb pieces mean"])
    nb_pieces.insert(0, 19.15)
    label = "every-visit,gamma=" if file.__contains__("every-visit") else "first-visit, gamma="
    label += ('0.95' if file.__contains__("0.95") else '0.75')
    plt.plot(ep_trained, nb_pieces, marker='x', label=label)

plt.legend()
plt.ylabel("Number of pieces placed")
plt.xlabel("Number of episodes trained (x10,000)")
plt.title("Learning curves for some Monte Carlo agents in extended fourer")
x_ticks = np.linspace(start=0, stop=20, num=11, dtype=int)
x_pos = np.linspace(start=0, stop=200000, num=11)
plt.xticks(x_pos, x_ticks)
plt.grid(axis='y', linestyle='--', lw=0.5)
plt.xlim(left=0, right=200000)
plt.show()
