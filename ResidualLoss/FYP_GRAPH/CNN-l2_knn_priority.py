import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = [
    "100",
    "50",
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
method_1 = [
    37623,
    37627,
    37600,
    37500,
    37450,
    37425,
    37400,
    37350,
    37200,
    37100,
    37050,
    37075
]
method_2 = [
    37623,
    37627,
    37624,
    37626,
    37400,
    37300,
    37000,
    36875,
    36500,
    36450,
    36250,
    36125
]


# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, baseline, color='black')
plt.plot(alpha_value_list, method_1, color='blue')
plt.plot(alpha_value_list, method_2, color='red')
# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['baseline', 'Distance-Based KNN', 'Multiplication-Based KNN', 'fixed reference single', 'fixed reference double'])

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(36000, 38000)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
