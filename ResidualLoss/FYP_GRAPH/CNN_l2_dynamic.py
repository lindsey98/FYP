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

baseline = [37517] * len(alpha_value_list)
method_original = [
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

method_best_loss = [
    37643.72,
    37633.945,
    37632.895,
    37652.205,
    37631.155,
    37625.33,
    37629.905,
    37624.695,
    37622.745,
    37626.05,
]

method_replace = [
    36626.525,
    37092.855,
    37557.04,
    37598.73,
    37621.335,
    37625.68,
    37621.715,
    37625.275,
    37625.07,
    37625.6,
]

method_single = [
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

# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, baseline, color='black')
plt.plot(alpha_value_list, method_original, color='blue')
plt.plot(alpha_value_list, method_best_loss, color='orange')
plt.plot(alpha_value_list, method_replace, color='purple')
plt.plot(alpha_value_list, method_single, color='red')

plt.legend(labels=['baseline', 'Original L2 with both part', 'Dynamic L2 with lower loss', 'Dynamic L2 with new reference', 'Original L2 with preserving part'])

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(37500, 37800)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
