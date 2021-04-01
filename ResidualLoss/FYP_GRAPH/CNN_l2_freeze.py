import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = [str(i) for i in [
    100,
    75,
    50,
    25,
    10,
    5,
    1,
    0.5,
    0.1,
    0.05,
    0.01,
]]

normal = [
    37557.665,
    37562.91,
    37563.42,
    37591.905,
    37567.56,
    37574.805,
    37574.14,
    37571.215,
    37574.65,
    37579.22,
    37549.015,
]

freeze = [
    37576.105,
    37579.95,
    37579.455,
    37577.655,
    37599.77,
    37583.205,
    37641.485,
    37649.065,
    37679.755,
    37688.725,
    37655.13,
]

baseline = [37517] * len(alpha_value_list)
freeze_baseline = [37623] * len(alpha_value_list)

plt.plot(alpha_value_list, baseline, color='black')
plt.plot(alpha_value_list, freeze_baseline, color='gray')
plt.plot(alpha_value_list, normal, color='blue')
plt.plot(alpha_value_list, freeze, color='red')
# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['baseline', 'freeze baseline', 'normal with l2 loss', 'freeze with l2 loss'])

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# plt.ylim(36000, 38000)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
