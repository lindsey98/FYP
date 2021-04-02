import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

lst = []
for i in range(6):
    start = 5 * i + 1
    end = 5 * i + 5

    f = open("../log/CNN_30_%s-%s.txt" % (start, end))
    lines = f.readlines()
    data = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]
    for j in range(490, 2501, 500):
        lst.append(int(sum(data[j: j + 10]) / 10))

plt.bar(range(1, 31), lst)
plt.plot(range(1, 31), [int(sum(lst) / 30)] * 30, color="red")
plt.ylim(33000, 36000)
plt.show()

