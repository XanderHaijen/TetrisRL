import os
import pandas as pd
from statistics import mean as statmean

upper_dir = r"D:\Bibliotheken\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData"
as_means, as_upper_err, as_lower_err, as_names = [], [], [], []
sa_means, sa_upper_err, sa_lower_err, sa_names = [], [], [], []
sa1_means, sa1_upper_err, sa1_lower_err, sa1_names = [], [], [], []

for dir in os.listdir(upper_dir):
    print(dir)
    if not dir.lower().__contains__("learning") and not dir.__contains__("fourer_SarsaZeroAfterstates") and \
            not dir.__contains__('SV'):
        means = set()
        lower_qs = set()
        upper_qs = set()
        if os.path.isdir(os.path.join(upper_dir, dir, "Results")):
            files = {csv for csv in os.listdir(os.path.join(upper_dir, dir, "Results")) if csv.__contains__(".csv")}
        else:
            files = {csv for csv in os.listdir(os.path.join(upper_dir, dir)) if csv.__contains__(".csv")}
        for csv in files:
            name = os.path.join(upper_dir, dir, "Results", csv)
            metrics = pd.read_csv(name)
            mean = metrics["Lines_cleared"].mean()
            lower_q = metrics["Lines_cleared"].quantile([0.25])
            upper_q = metrics["Lines_cleared"].quantile([0.75])

            means.update({mean})
            lower_qs.update({float(lower_q)})
            upper_qs.update({float(upper_q)})

        avg = sum(means) / len(means)
        lower_q = sum(lower_qs) / len(lower_qs)
        upper_q = sum(upper_qs) / len(upper_qs)

        if dir.lower().__contains__("mc"):
            as_names.append(dir)
            as_means.append(avg)
            as_upper_err.append(upper_q - avg)
            as_lower_err.append(avg - lower_q)
        elif dir.__contains__("0"):
            sa_names.append(dir)
            sa_means.append(avg)
            sa_upper_err.append(upper_q - avg)
            sa_lower_err.append(avg - lower_q)
        else:
            sa1_names.append(dir)
            sa1_means.append(avg)
            sa1_upper_err.append(upper_q - avg)
            sa1_lower_err.append(avg - lower_q)

print("-------------------------")
print(as_names, as_means, as_upper_err, as_lower_err, sep='\n')
print("--------------------------")
print(sa_names, sa_means, sa_upper_err, sa_lower_err, sep='\n')
print('-------------------------')
print(sa1_names, sa1_means, sa1_upper_err, sa1_lower_err, sep='\n')
