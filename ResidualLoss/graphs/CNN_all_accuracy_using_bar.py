import matplotlib.pyplot as plt

from ResidualLoss.graphs.CNN_data import *

plt.title("All Accuracy Max Bar Graph")

name_list = ['alpha = 0', '1', '2', '12', '123']

num_list = [
    max(y_ref_acc_1),
    max(y_acc_1_1),
    max(y_acc_2_1),
    max(y_acc_12_1),
    max(y_acc_123_1),
]

plt.ylim(37500, 37600)
plt.bar(range(len(num_list)), num_list, tick_label=name_list)
plt.show()
