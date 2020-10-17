import matplotlib.pyplot as plt

from ResidualLoss.graphs.MLP_data import *

plt.title("All Accuracy Max Bar Graph")

name_list = ['alpha = 0', '1', '2', '3', '12', '13', '23', '123', '1234', 'Freeze 1']

num_list = [max(y_ref_acc_1),
            max(y_acc_1_1), max(y_acc_2_1), max(y_acc_3_1),
            max(y_acc_12_1), max(y_acc_13_1), max(y_acc_23_1), max(y_acc_123_1), max(y_acc_1234_1),
            max(y_freeze_1_acc_1)]

plt.ylim(42000, 43000)
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
plt.show()
