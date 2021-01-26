import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = range(200)

f = open("../log/CNN_l2_freeze_prob_far_dist.txt")
lines = f.readlines()
data_freeze = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]

f = open("../log/CNN_l2_prob_far_dist.txt")
lines = f.readlines()
data_unfreeze = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]

f = open("../log/CNN_l2_prob_far_dist_reinit.txt")
lines = f.readlines()
data_unfreeze_re_init = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]

print(len(data_freeze), len(data_unfreeze), len(data_unfreeze_re_init))


# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, data_freeze, color='orange')
plt.plot(alpha_value_list, data_unfreeze, color='red')
plt.plot(alpha_value_list, data_unfreeze_re_init, color='blue')
plt.plot(alpha_value_list, [37623] * 200, color='black')


plt.title("Three result using different methods")

plt.legend(labels=['freeze', 'unfreeze', 'unfreeze_re_init', 'baseline'])

plt.xlabel("epoch")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(25000, 38000)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))


plt.show()