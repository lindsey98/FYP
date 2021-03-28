import matplotlib.pyplot as plt
from matplotlib import ticker

alpha_value_list_for_MLP = [str(i) for i in [
    0.5,
    0.1,
    0.05,
    0.01,
    0.005,
    0.001,
]]

cosine_similarity_MLP = [
    41746.165,
    42781.89,
    42805.185,
    42626.48,
    42534.865,
    42455.95,
]

l2_MLP = [
    42363.875,
    42614.365,
    42757.9,
    42872.3,
    42765.945,
    42536.98,
]

baseline_MLP = [42416] * len(alpha_value_list_for_MLP)

plt.plot(alpha_value_list_for_MLP, baseline_MLP, color='black')
plt.plot(alpha_value_list_for_MLP, cosine_similarity_MLP, color='blue')
plt.plot(alpha_value_list_for_MLP, l2_MLP, color='red')
# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['baseline', 'layer 1 with cosine similarity loss', 'layer 1 with l2 loss'])

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# plt.ylim(36000, 38000)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()

alpha_value_list_for_CNN = [str(i) for i in [
    1000,
    500,
    200,
    100,
    75,
    50,
    25,
    10,
    5,
]]

baseline_CNN = [37517] * len(alpha_value_list_for_CNN)

cosine_similarity_CNN = [
    37571.395,
    37580.32,
    37578.67,
    37587.06,
    37580.485,
    37587.875,
    37586.26,
    37577.325,
    37575.41,
]

l2_CNN = [
    37567.445,
    37557.77,
    37564.625,
    37557.665,
    37562.91,
    37563.42,
    37591.905,
    37567.56,
    37574.805,
]

# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.plot(alpha_value_list_for_CNN, baseline_CNN, color='black')
plt.plot(alpha_value_list_for_CNN, cosine_similarity_CNN, color='blue')
plt.plot(alpha_value_list_for_CNN, l2_CNN, color='red')
# plt.title("All Accuracy Using Seed 1")

plt.legend(labels=['baseline', 'layer 3 with cosine similarity loss', 'layer 3 with l2 loss'],
           loc=(0.4, 0.1))

plt.xlabel("α")
plt.ylabel("Training Accuracy")
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# plt.ylim(36000, 38000)


def to_percent(temp, position):
    return '%1.2f' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
