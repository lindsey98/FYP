import matplotlib.pyplot as plt

from ResidualLoss.data import alpha_value_1, y_loss_1_1, y_acc_1234_1, y_acc_1234_2, y_acc_123_1, y_acc_123_2

ln1, = plt.plot(alpha_value_1, y_acc_1234_1, color='red')
ln2, = plt.plot(alpha_value_1, y_acc_1234_2, color='red', linestyle='-.')
ln3, = plt.plot(alpha_value_1, y_acc_123_1, color='blue')
ln4, = plt.plot(alpha_value_1, y_acc_123_2, color='blue', linestyle='-.')

plt.title("Using Last Layer Or Not")

plt.legend(handles=[ln1, ln2, ln3, ln4], labels=['Layer 1234 Seed 1', 'Layer 1234 Seed 2', 'Layer 123 Seed 1', 'Layer 123 Seed 2'])

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()