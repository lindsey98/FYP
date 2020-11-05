import matplotlib.pyplot as plt
from matplotlib import ticker

from ResidualLoss.graphs.CNN_data import *

plt.plot(alpha_value_2, y_ref_acc_1, color='black', label='alpha = 0')
ln1 = plt.plot(alpha_value_2, y_acc_1_1, color='red')
ln2 = plt.plot(alpha_value_2, y_acc_2_1, color='tomato')
ln3 = plt.plot(alpha_value_2, y_acc_3_1, color='chocolate')
ln4 = plt.plot(alpha_value_2, y_acc_12_1, color='yellow')
ln5 = plt.plot(alpha_value_2, y_acc_13_1, color='green')
ln6 = plt.plot(alpha_value_2, y_acc_23_1, color='blue')
ln7 = plt.plot(alpha_value_2, y_acc_123_1, color='purple')

# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['alpha = 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 12', 'Layer 13', 'Layer 23', 'Layer 123'],
           bbox_to_anchor=(0.97, 1), loc='upper left', borderaxespad=0.)

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(37510, 37600)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))


plt.show()