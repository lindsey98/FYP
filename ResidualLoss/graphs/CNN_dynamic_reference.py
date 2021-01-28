import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = [
    "10",
    "5",
    "1",
    "0.5",
    "0.1",
    "0.05",
    "0.01",
    "0.005",
    "0.001",
    "0.0005",
]

baseline = [37623] * len(alpha_value_list)
current_best = [37938] * len(alpha_value_list)
method_1 = [
    36529.745,
    37048.51,
    37561.48,
    37597.425,
    37625.06,
    37625.95,
    37626.58,
    37624.36,
    37623.375,
    37624.4,
]

method_2 = [
    37646.635,
    37632.74,
    37640.655,
    37649.01,
    37630.285,
    37624.01,
    37624.01,
    37624.01,
    37624.01,
    37624.01,
]

method_3 = [
    37625.045,
    37625.89,
    37629.7,
    37626.71,
    37624.185,
    37628.55,
    37653.02,
    37647.675,
    37633.475,
    37630.975,
]

method_4 = [
    37599.77,
    37583.205,
    37641.485,
    37649.065,
    37679.755,
    37688.725,
    37655.13,
    37643.46,
    37629.865,
    37628.95,
]

# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, baseline, color='black')
plt.plot(alpha_value_list, current_best, color='gray')
plt.plot(alpha_value_list, method_1, color='blue')
plt.plot(alpha_value_list, method_2, color='orange')
plt.plot(alpha_value_list, method_3, color='purple')
plt.plot(alpha_value_list, method_4, color='red')
# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['baseline', 'current_best', 'dynamic reference replace', 'dynamic reference best loss', 'fixed reference single', 'fixed reference double'])

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(37200, 37700)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
