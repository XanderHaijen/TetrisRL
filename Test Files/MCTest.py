import math

from Models.OnPolicyMCForTetris import OnPolicyMCForTetris
from Evaluation.Evaluate_policy import evaluate_policy
import matplotlib.pyplot as plt


def on_policy_mc_test(first_visit: bool, training_batch_size: int = 1000, nb_batches: int = 3, gamma: float = 1):
    model = OnPolicyMCForTetris(first_visit=first_visit)
    episodes_trained = 0
    scores, nbs_pieces, episodes = [], [], []
    for i in range(1, 1 + nb_batches):
        print(f'Round {i}')
        model.train(learning_rate=lambda x: 1 / (10 + math.log(1 + x)), nb_episodes=training_batch_size,
                    start_episode=episodes_trained, gamma=gamma)
        episodes_trained += training_batch_size
        metric = evaluate_policy(model, model.env, 50)
        episodes.append(episodes_trained)
        scores.append((metric.mean()['Score'], metric.var()['Score']))
        nbs_pieces.append((metric.mean()['Nb_pieces'], metric.var()['Nb_pieces']))

    model.save(r"D:\Bibliotheken\Downloads\MC.pickle")

    plt.plot(episodes, [data[0] for data in scores], 'r', episodes, [data[0] for data in nbs_pieces], 'b')
    plt.ylabel('Average score / nb_pieces')
    plt.xlabel('nb episodes trained')
    plt.savefig(r"D:\Bibliotheken\Downloads\plot.png", dpi=300, bbox_inches='tight')
    plt.close()


on_policy_mc_test(first_visit=False, training_batch_size=100, nb_batches=5, gamma=0.5)
