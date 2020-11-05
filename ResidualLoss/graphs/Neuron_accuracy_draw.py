import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
accuracy = [95.3, 95.8, 94.4, 96.8, 96.1, 95.5, 95.4, 95.9, 96.6, 95.3]
accuracy_1 = [95.3, 92.1, 87.5, 84.9, 81.3, 86.6, 81.7, 82.7, 79.6, 81.7]
accuracy_2 = [95.3, 97.5, 97.1, 97.2, 97.2, 97.7, 97.9, 97.7, 97.2, 97.9]
accuracy_3 = [95.3, 93.6, 91.0, 88.7, 88.4, 88.8, 88.2, 88.4, 88.0, 88.8]

plt.plot(epoch, accuracy, color='black')
ln1 = plt.plot(epoch, accuracy_1, color='red')
ln2 = plt.plot(epoch, accuracy_2, color='blue')
ln4 = plt.plot(epoch, accuracy_3, color='orange')

plt.legend(labels=['normal retraining', 'neuron-level-freezing retraining', 'layer-level-freezing retraining', 'random dropout retraining'])
plt.ylim(75, 100)
ax = plt.gca()


def to_percent(temp, position):
    return '%1.0f' % temp + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()
