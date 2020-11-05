import matplotlib.pyplot as plt

from ResidualLoss.graphs.CNN_data import *

plt.plot(alpha_value_2, y_ref_acc_1, color='black', label='alpha = 0')
ln1 = plt.plot(alpha_value_2, y_acc_123_1, color='red')
ln2 = plt.plot(alpha_value_2, y_acc_123_1_not_loaded, color='firebrick')
ln3 = plt.plot(alpha_value_2, y_acc_23_1, color='deepskyblue')
ln4 = plt.plot(alpha_value_2, y_acc_23_1_not_loaded, color='dodgerblue')
ln5 = plt.plot(alpha_value_2, y_acc_3_1, color='yellow')
ln6 = plt.plot(alpha_value_2, y_acc_3_1_not_loaded, color='orange')

plt.title("Difference of re-loading ref model or not")

plt.legend(labels=['alpha = 0', 'layer 123 L', 'layer 123 N', 'layer 23 L', 'layer 23 N', 'layer 3 L', 'layer 3 N'],
           bbox_to_anchor=(0.97, 1), loc='upper left', borderaxespad=0.)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()