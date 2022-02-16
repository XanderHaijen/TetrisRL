import matplotlib.pyplot as plt
from Reporting.MC_plots import displacement
names = ['Fourer', "Extended fourer"]
mc_means = [9.12434, 3.660081]
mc_upper_err = [3.9526, 1.3399]
mc_lower_err = [6.3243, 2.660087]

s0_means = [44.82, 9.0944]
s0_upper_err = [23.8159, 2.9055]
s0_lower_err = [32.55421, 4.0944625]

sl_means = [19.08965, 6.714733]
sl_upper_err = [19.0865, 2.03526666]
sl_lower_err = [11.3118, 3.71473333]

x_axis = list(range(len(names)))

i = 1
x_axis_temp = [1 + x + displacement(i, 3, width=0.03) for x in x_axis]
plt.errorbar(x_axis_temp, s0_means, yerr=(s0_lower_err, s0_upper_err), label="Sarsa(0)", fmt='x', linestyle='--')

i = 2
x_axis_temp = [1 + x + displacement(i, 3, width=0.03) for x in x_axis]
plt.errorbar(x_axis_temp, mc_means, yerr=(mc_lower_err, mc_upper_err), label="Monte Carlo", fmt='x', linestyle='--')

i = 3
x_axis_temp = [1 + x + displacement(i, 3, width=0.03) for x in x_axis]
plt.errorbar(x_axis_temp, sl_means, yerr=(sl_lower_err, sl_upper_err), label="Sarsa(lambda)", fmt='x', linestyle='--')

plt.xticks([1 + x for x in range(len(names))], names)
plt.grid(axis='y', linestyle='--', lw=0.5)
# plt.xlim(right=3.15)
plt.ylim(bottom=0)
plt.title("Fourer versus Extended fourer")
plt.ylabel("Lines Cleared")
plt.legend()

plt.show()
