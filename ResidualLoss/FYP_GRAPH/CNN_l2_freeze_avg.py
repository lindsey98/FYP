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
method_GAP = [
    37567.41,
    37579.655,
    37582.765,
    37577.175,
    37591.665,
    37577.02,
    37548.815,
    37529.385,
    37522.62,
    37522.9,
]

method_2 = [
    37571.895,
    37566.345,
    37574.01,
    37572.48,
    37574.87,
    37577.29,
    37551.635,
    37531.02,
    37522.76,
    37522.65,
]

method_flatten = [
    37567.88,
    37597.07,
    37573.12,
    37572.64,
    37577.02,
    37580.67,
    37551.47,
    37531.96,
    37523.035,
    37523.955,
]

# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, baseline, color='black')
plt.plot(alpha_value_list, method_GAP, color='blue')
plt.plot(alpha_value_list, method_2, color='orange')
plt.plot(alpha_value_list, method_flatten, color='purple')
# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['baseline', 'GAP', 'Flatten after average pooling', 'Flatten only'])

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(37500, 37650)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
