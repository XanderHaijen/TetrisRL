import matplotlib.pyplot as plt
from Reporting.MC_plots import displacement
sa_names = ['Monte Carlo', 'Sarsa(0)', 'Sarsa(lambda)']
as_means = [ 9.12434, 44.8269375, 19.089650000000002]
as_upper_err = [ 3.9525830769230765, 23.81591964285714, 7.852657692307691]
as_lower_err = [ 6.32434, 32.554210227272726, 11.311872222222224]

sa_means = [0.8811888888888888, 1.0363235294117648, 0.9835583333333333]
sa_upper_err = [0.1188111111111112, 0.7761764705882352, 0.5164416666666667]
sa_lower_err = [0.8811888888888888, 1.0363235294117648, 0.9835583333333333]

i = 1
x_axis = list(range(len(sa_names)))

x_axis_temp = [1 + x + displacement(i, 2, width=0.075) for x in x_axis]

plt.errorbar(x_axis_temp, sa_means, yerr=(sa_lower_err, sa_upper_err), label="State-action agents", fmt='x')
i = 2
x_axis_temp = [1 + x + displacement(i, 2, width=0.075) for x in x_axis]

plt.errorbar(x_axis_temp, as_means, yerr=(as_lower_err, as_upper_err), label="Afterstate agents", fmt='x')

plt.xticks([1 + x for x in range(len(sa_names))], sa_names)
plt.grid(axis='y', linestyle='--', lw=0.5)
plt.xlim(right=3.15)
plt.title("State-action versus afterstate agents")
plt.ylabel("Lines Cleared")
plt.legend()
plt.show()