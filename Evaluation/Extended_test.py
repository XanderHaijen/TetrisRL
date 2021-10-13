import os
from typing import Type



from Evaluation.Evaluate_policy import evaluate_policy_afterstates
from Models.AfterstateModel import AfterstateModel
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates


def extended_test(model_type: Type[AfterstateModel], models_dir):
    text_file = os.path.join(target_dir, "results.txt")
    f = open(text_file, "w+")
    for model_dir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir, "Model", "model.pickle")
        model = model_type.load(model_path)
        metrics = evaluate_policy_afterstates(model, model.env, 2000)
        results_dir = os.path.join(target_dir, "Results")

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        f.write(f"\n \n \n For {model}: \n \n ")
        f.write("Mean and quantiles: \n")
        f.write(str(metrics.mean()))
        f.write("\n")
        f.write(str(metrics.quantile([0.25, 0.5, 0.75])))
        f.write("\n")
        f.write("Variance: \n")
        f.write(str(metrics.var()))
        f.write("\n Maximum and minimum \n ")
        f.write(str(metrics.max()))
        f.write(str(metrics.min()))
        f.write("\n ------------------------------------------------------------"
                " \n ------------------------------------------------------------ \n ")
        metrics.to_csv(os.path.join(results_dir, f"{model}.csv"))

    f.write("The end")


models_dir = "scratch/leuven/343/vsc34339/RLData/fourer_sarsa"
target_dir = "data/leuven/343/vsc34339/RLData"
model_type = SarsaZeroAfterStates
extended_test(model_type, models_dir)
