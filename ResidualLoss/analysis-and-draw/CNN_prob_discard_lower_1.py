import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = range(200)

f = open("../log/CNN_prob_discard_lower_1.txt")
lines = f.readlines()
data_train = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]

# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, data_train, color='orange')
plt.plot(alpha_value_list, [37618.39] * 200, color='red')
plt.plot(alpha_value_list, [37517] * 200, color='black')


# plt.title("Three result using different methods")

plt.legend(labels=['train', 'average', 'baseline_unfreeze'])

plt.xlabel("epoch")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(37200, 37800)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))


plt.show()