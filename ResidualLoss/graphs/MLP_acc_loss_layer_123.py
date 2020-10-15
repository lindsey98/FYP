import matplotlib.pyplot as plt

from ResidualLoss.MLP_data import *

fig, ax1 = plt.subplots()

color = 'red'
ax1.set_xlabel('alpha value')
ax1.set_ylabel('accuracy', color=color)
ax1.plot(alpha_value_2, y_acc_123_1, color=color)
ax1.plot(alpha_value_2, y_acc_123_2, color='pink')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'blue'
ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
ax2.plot(alpha_value_2, y_loss_123_1, color=color)
ax2.plot(alpha_value_2, y_loss_123_2, color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()