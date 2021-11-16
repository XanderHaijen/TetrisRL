import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from Models.SarsaLambdaAfterstates import SarsaLambdaAfterstates
from Evaluation.Render_policy import render_policy_afterstates
from Evaluation.Evaluate_policy import evaluate_policy_afterstates

file = r"C:\Users\xande\Downloads\model.pickle"
model = SarsaLambdaAfterstates.load(file, False)
metrics = evaluate_policy_afterstates(model, model.env, 1000)
print(metrics.mean())
print(metrics.quantile([0.25, 0.75]))

# dir = r"C:\Users\xande\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData\MC_fourer_AS_learningCurves"
# for file in os.listdir(dir):
#     path = os.path.join(dir, file)
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#         f.close()
#     ep_trained = data[0]
#     ep_trained.insert(0,0)
#     nb_pieces = [x for x,_ in data[2]]
#     nb_pieces.insert(0, 24.56)
#     label = "every-visit,gamma=" if file.__contains__("every-visit") else "first-visit, gamma="
#     label += ('0.9' if file.__contains__("0.9") else '0.7')
#     plt.plot(ep_trained, nb_pieces, marker='x', label=label)
#
# plt.legend()
# plt.ylabel("Number of pieces placed")
# plt.xlabel("Number of episodes trained (x10,000)")
# plt.title("Learning curves for some Monte Carlo agents in fourer")
# x_ticks = np.linspace(start=0, stop=20, num=11, dtype=int)
# x_pos = np.linspace(start=0, stop=200000, num=11)
# plt.xticks(x_pos, x_ticks)
# plt.grid(axis='y', linestyle='--', lw=0.5)
# plt.xlim(left=0, right=200000)
# plt.show()
