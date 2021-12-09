import os
import pandas as pd

file = r"C:\Users\xande\OneDrive\Documenten\KULeuven\ReinforcementLearning\RLData"
as_means, as_upper_err, as_lower_err, as_names = [], [], [], []
sa_means, sa_upper_err, sa_lower_err, sa_names = [], [], [], []
for dir in os.listdir(file):
    print(dir)
    if not dir.lower().__contains__("learning"):
        means = set()
        lower_qs = set()
        upper_qs = set()
        if os.path.isdir(os.path.join(file, dir, "Results")):
            files = {csv for csv in os.listdir(os.path.join(file, dir, "Results")) if csv.__contains__(".csv")}
        else:
            files = {csv for csv in os.listdir(os.path.join(file, dir)) if csv.__contains__(".csv")}
        for csv in files:
            name = os.path.join(file, dir, "Results", csv)
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

        if dir.lower().__contains__("as"):
            as_names.append(dir)
            as_means.append(avg)
            as_upper_err.append(upper_q - avg)
            as_lower_err.append(avg - lower_q)
        else:
            sa_names.append(dir)
            sa_means.append(avg)
            sa_upper_err.append(upper_q - avg)
            sa_lower_err.append(avg - lower_q)

print(as_names)
print(as_means)
print(as_upper_err)
print(as_lower_err)
print("______________________")
print(sa_names)
print(sa_means)
print(sa_upper_err)
print(sa_lower_err)

