import matplotlib.pyplot as plt


epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
accuracy = [95.34, 95.73, 94.95, 96.98, 96.13, 95.31, 95.83, 95.77, 96.61, 95.99]
accuracy_1 = [95.34, 92.73, 87.95, 84.98, 81.13, 81.31, 81.83, 81.77, 81.61, 81.99]
accuracy_2 = [95.34, 97.73, 97.95, 97.98, 97.13, 97.31, 97.83, 97.77, 97.61, 97.99]
accuracy_3 = [95.34, 92.73, 87.95, 84.98, 81.13, 81.31, 81.83, 81.77, 81.61, 81.99]

plt.plot(epoch, accuracy, color='black')
ln1 = plt.plot(epoch, accuracy_1, color='red')
ln2 = plt.plot(epoch, accuracy_2, color='blue')
ln4 = plt.plot(epoch, accuracy_3, color='yellow')

plt.legend(labels=['normal retraining', 'neuron level freezing retraining', 'layer level freezing retraining', 'random dropout retraining'])

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()