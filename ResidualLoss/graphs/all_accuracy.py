import matplotlib.pyplot as plt

from ResidualLoss.data import *

plt.plot(alpha_value_2, y_ref_acc_1, color='black', label='alpha = 0')
ln1 = plt.plot(alpha_value_2, y_acc_1_1, color='red')
ln2 = plt.plot(alpha_value_2, y_acc_2_1, color='tomato')
ln3 = plt.plot(alpha_value_2, y_acc_3_1, color='chocolate')
ln4 = plt.plot(alpha_value_2, y_acc_12_1, color='yellow')
ln5 = plt.plot(alpha_value_2, y_acc_13_1, color='green')
ln6 = plt.plot(alpha_value_2, y_acc_23_1, color='blue')
ln7 = plt.plot(alpha_value_2, y_acc_123_1, color='purple')
ln8 = plt.plot(alpha_value_2, y_acc_1234_1, color='midnightblue')
ln9 = plt.plot(alpha_value_2, y_freeze_1_acc_1, color='grey')

plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['alpha = 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 12', 'Layer 13', 'Layer 23', 'Layer 123', 'Layer 1234', 'Freeze 1'],
           bbox_to_anchor=(0.97, 1), loc='upper left', borderaxespad=0.)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()