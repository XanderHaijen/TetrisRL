from Models.SarsaLambdaForTetris import SarsaLambdaForTetris
from Evaluation import Evaluate_policy
from math import log

from Models.SarsaZeroForTetris import SarsaZeroForTetris


def sarsa_zero_test():
    model = SarsaLambdaForTetris(0.9, 0.1, 0.2, "replacing")
    episodes_trained = 0
    for i in range(3):
        print(f"Training round {i + 1}")
        model.train(learning_rate=lambda x: 1 / (10 + log(1+x)), nb_episodes=100, start_episode=episodes_trained)
        episodes_trained += 1000
        print(f"Evaluation round {i + 1}")
        metrics = Evaluate_policy.evaluate_policy(model, model.env, nb_episodes=50)
        print(metrics.mean())
        print(metrics.var())
    model.save(r"D:\Bibliotheken\Downloads\model.pickle")


def render_view():
    model = SarsaLambdaForTetris.load(r"D:\Bibliotheken\Downloads\model.pickle")
    Evaluate_policy.evaluate_policy(model, model.env, nb_episodes=20, render=True)


render_view()