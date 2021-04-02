import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import ticker

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

alpha_value_list = [str(i) for i in [
    1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
]]

baseline = [37517] * len(alpha_value_list)

color = []
labels = []
for i in range(1, 11):
    eta = 0.05 * i
    if i % 2 == 1:
        loc = "../log/CNN_l2_freeze_prob_%.2f_%.2f.txt" % (1 - eta, 1 + eta)
        labels.append("β-%.2f" % eta)
    else:
        loc = "../log/CNN_l2_freeze_prob_%.1f_%.1f.txt" % (1 - eta, 1 + eta)
        labels.append("β-%.1f" % eta)

    f = open(loc)
    lst = []
    lines = f.readlines()
    for j in range(11):
        line = lines[8 * j: 8 * j + 8]
        lst.append(float(line[-3].split("average correct: ")[-1].split("\n")[0]))
        # data = [ for i in lines if "epoch" in i]

    plt.plot(alpha_value_list, lst)

plt.plot(alpha_value_list, baseline, color="black")
labels.append("baseline")

plt.legend(labels=labels,
           prop={'size': 8},
           loc=(0.8, 0.08))

plt.xlabel("γ")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# plt.ylim(37500, 37800)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()

