import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list = [str(i) for i in [
    1,
    1.5,
    2,
    4,
    6,
    8,
    10,
]]

baseline = [37517] * len(alpha_value_list)

data = [
    [
        37625.045,
        37626.505,
        37628.04,
        37624.01,
        37617.69,
        37613.105,
        37607.965,
    ], [
        37625.89,
        37628.92,
        37626.785,
        37615.215,
        37606.825,
        37594.995,
        37578.23,
    ], [
        37641.485,
        37617.15,
        37607.445,
        37540.785,
        37445.21,
        37364.335,
        37301.17,
    ], [
        37649.065,
        37608.305,
        37590.565,
        37453.235,
        37346.155,
        37278.53,
        37235.76,
    ], [
        37679.755,
        37606.41,
        37567.78,
        37412.785,
        37280.305,
        37105.48,
        36900.455,
    ], [
        37688.725,
        37613.665,
        37579.135,
        37382.53,
        37136.745,
        36872.905,
        36627.1,
    ], [
        37655.13,
        37617.1,
        37556.535,
        37208.93,
        36824.075,
        36506.95,
        36215.045,
    ]
]

labels = ["α-%s" % i for i in [
    10,
    5,
    1,
    0.5,
    0.1,
    0.05,
    0.01
]]
labels.append("baseline")
# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

for lst in data:
    plt.plot(alpha_value_list, lst)
plt.plot(alpha_value_list, baseline, color='black')

plt.legend(labels=labels)

plt.xlabel("γ")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.ylim(36000, 38000)


def to_percent(temp, position):
    return '%d' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
