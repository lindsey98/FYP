f = open("./log/CNN_CIFAR_11_6_X_7.txt")
lines = f.readlines()
data = [int(i.split("Accuracy: ")[-1].split("/50000\n")[0]) for i in lines if "epoch" in i]
for i in range(4990, 15001, 5000):
    print(int(sum(data[i: i + 10]) / 10))