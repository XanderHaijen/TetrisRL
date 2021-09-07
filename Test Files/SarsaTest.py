from Algorithms import SarsaZeroForTetris
from Evaluation import Evaluate_policy
from math import log


def model_test():
    model = SarsaZeroForTetris.SarsaZeroForTetris(alpha=0.01, gamma=0.01)
    episodes_trained = 0
    for i in range(3):
        print(f"Training round {i + 1}")
        model.train(learning_rate=lambda x: 1 / (10 + log(1+x)), nb_episodes=1000, start_episode=episodes_trained)
        episodes_trained += 1000
        print(f"Evaluation round {i + 1}")
        metrics = Evaluate_policy.evaluate_policy(model, model.env, nb_episodes=1000)
        print(metrics.mean())
        print(metrics.var())
    model.save(r"D:\Bibliotheken\Downloads\model.pickle")


def render_view():
    model = SarsaZeroForTetris.SarsaZeroForTetris.load(r"D:\Bibliotheken\Downloads\model.pickle")
    Evaluate_policy.evaluate_policy(model, model.env, nb_episodes=20, render=True)


model_test()
