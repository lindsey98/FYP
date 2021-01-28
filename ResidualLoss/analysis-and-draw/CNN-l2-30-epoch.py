import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = range(1800)

color_list = ["red", "orange", "blue", "green", "purple"]
for i in range(5):
    f = open("../log/CNN_l2_30_%s-%s.txt" % (6 * i + 1, 6 * i + 6))
    lines = f.readlines()
    data = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]
    plt.plot(alpha_value_list, data, color=color_list[i])

plt.plot(alpha_value_list, [37517] * 1800, color='black')

plt.title("5 time training")

# plt.legend(labels=['freeze', 'unfreeze', 'unfreeze_re_init', 'baseline'])

plt.rcParams['figure.figsize'] = (10.0, 8.0)
# plt.rcParams['figure.dpi'] = 100

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
