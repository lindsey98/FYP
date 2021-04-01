import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

f = open("./data/3_round.txt")
lines = f.readlines()
data = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0].split(": ")[1]) for i in lines if "epoch" in i]

# plt.rcParams['figure.figsize'] = (10.0, 6.0)

plt.plot(range(30), data)

lst = []
for i in range(10):
    lst.append(None)
    lst.append("E%s" % (i + 1))
    lst.append(None)

    plt.scatter(range(3 * i, 3 * i + 3), data[3 * i: 3 * i + 3])

plt.xticks(range(30), lst)
# plt.xlabel('Learnable Rate')
# plt.ylabel('Percentage')
# plt.legend(loc='upper left')
# plt.xlim(-0.5, 31)
ax = plt.gca()


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
