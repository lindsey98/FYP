import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = [
    "10",
    "5",
    "1",
    "0.5",
    "0.1",
    "0.05",
    "0.01",
    "0.005",
    "0.001",
    "0.0005",
]

baseline = [37517] * len(alpha_value_list)

raw_method = [
    37599.77,
    37583.205,
    37641.485,
    37649.065,
    37679.755,
    37688.725,
    37655.13,
    37643.46,
    37629.865,
    37628.95,
]

kk1_l1 = [
    37655.805,
    37672.805,
    37659.525,
    37654.43,
    37633.945,
    37627.555,
    37627.765,
    37625.79,
    37625.33,
    37624.005,
]

kk5_l2 = [
    37652.585,
    37659.145,
    37653.17,
    37644.1,
    37628.83,
    37626.88,
    37626.175,
    37625.15,
    37623.195,
    37625.795,
]

kk3_l2 = [
    37650.995,
    37682.37,
    37679,
    37659.535,
    37635.545,
    37628.965,
    37627.43,
    37625.7,
    37623.565,
    37626.71,
]

data = [
    8159.41,
    9925.065,
    37322.34,
    37566.75,
    37618.915,
    37621.36,
    37625.315,
    37624.545,
    37623.47,
    37625.69,
]

data_1 = [
    37666.005,
    37671.24,
    37653.285,
    37643.605,
    37628.68,
    37626.07,
    37626.29,
    37624.89,
    37622.39,
    37626.365,
]

plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list, baseline, color='black')
plt.plot(alpha_value_list, raw_method, color='blue')
plt.plot(alpha_value_list, data, color='orange')
plt.plot(alpha_value_list, kk1_l1, color='red')
plt.plot(alpha_value_list, kk3_l2, color='green')
plt.plot(alpha_value_list, data_1, color='purple')

plt.legend(labels=['baseline',
                   'unweighted',
                   'k=0',
                   'k=1',
                   'k=3',
                   'k=5'],
           loc=(0.68, 0.1))

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(37510, 37700)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))


plt.show()
