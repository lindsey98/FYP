import matplotlib.pyplot as plt

from ResidualLoss.graphs.CNN_data import *

plt.plot(alpha_value_2, y_ref_acc_1, color='black', label='alpha = 0')
ln1 = plt.plot(alpha_value_2, y_acc_1_1, color='red')
ln2 = plt.plot(alpha_value_2, y_acc_2_1, color='tomato')
ln4 = plt.plot(alpha_value_2, y_acc_12_1, color='yellow')
ln7 = plt.plot(alpha_value_2, y_acc_123_1, color='purple')

plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['alpha = 0', 'Layer 1', 'Layer 2', 'Layer 12', 'Layer 123'],
           bbox_to_anchor=(0.97, 1), loc='upper left', borderaxespad=0.)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()