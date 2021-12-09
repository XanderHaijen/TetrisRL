import os
from typing import Type

from Evaluation.Evaluate_policy import *
from Models.AfterstateModel import AfterstateModel


def extended_test(model_type: Type[AfterstateModel], models_dir: str,
                  target_dir: str = None, nb_episodes: int = 2000) -> None:
    """
    Runs an extensive test of all models of the same type in a given folder. For every model, a csv containing
    score, pieces placed and lines cleared is created. A text file containing an overview of all test is also created
    :param model_type: the type of model, must be subclass of AfterstateModel
    :param models_dir: the directory containing all the models to test
    :raises TypeError if not all models are of the type provided in model_type
    :param target_dir: the directory which is to contain all results
    :param nb_episodes: the amount of episodes each model will train
    :return: None
    """
    if target_dir is None:
        target_dir = models_dir

    text_file = os.path.join(target_dir, "results.txt")
    f = open(text_file, "w+")
    for model_dir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir, "Model", "model.pickle")
        model = model_type.load(model_path)
        metrics = evaluate_policy_afterstates(model, model.env, nb_episodes)
        results_dir = os.path.join(target_dir, "Results")

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        f.write(f"\n \n \n For {model}: \n \n ")
        f.write("Mean and quantiles: \n")
        f.write(str(metrics.mean()))
        f.write("\n")
        f.write(str(metrics.quantile([0.25, 0.5, 0.75])))
        f.write("\nVariance: \n")
        f.write(str(metrics.lower_q()))
        f.write("\n Maximum and minimum \n ")
        f.write(str(metrics.max()))
        print("\n")
        f.write(str(metrics.min()))
        f.write("\n ------------------------------------------------------------"
                "\n ------------------------------------------------------------\n ")
        metrics.to_csv(os.path.join(results_dir, f"{model}.csv"))

    f.write("The end")
    f.close()


def extended_test_state_action(model_type, models_dir: str,
                               target_dir: str = None, nb_episodes: int = 2000) -> None:
    """
    Runs an extensive test of all models of the same type in a given folder. For every model, a csv containing
    score, pieces placed and lines cleared is created. A text file containing an overview of all test is also created
    :param model_type: the type of model, must be subclass of StateValueModel
    :param models_dir: the directory containing all the models to test
    :raises TypeError if not all models are of the type provided in model_type
    :param target_dir: the directory which is to contain all results
    :param nb_episodes: the amount of episodes each model will train
    :return: None
    """
    if target_dir is None:
        target_dir = models_dir

    text_file = os.path.join(target_dir, "results.txt")
    f = open(text_file, "w+")
    for model_dir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir, "Model", "model.pickle")
        model = model_type.load(model_path)
        metrics = evaluate_policy_state_action(model, model.env, nb_episodes)
        results_dir = os.path.join(target_dir, "Results")

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        f.write(f"\n \n \n For {model}: \n \n ")
        f.write("Mean and quantiles: \n")
        f.write(str(metrics.mean()))
        f.write("\n")
        f.write(str(metrics.quantile([0.25, 0.5, 0.75])))
        f.write("\nVariance: \n")
        f.write(str(metrics.lower_q()))
        f.write("\n Maximum and minimum \n ")
        f.write(str(metrics.max()))
        print("\n")
        f.write(str(metrics.min()))
        f.write("\n ------------------------------------------------------------"
                "\n ------------------------------------------------------------\n ")
        metrics.to_csv(os.path.join(results_dir, f"{model}.csv"))

    f.write("The end")
    f.close()
