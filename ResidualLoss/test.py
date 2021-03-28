string = "epoch [4991/5000], loss:0.6460 Accuracy: 38665/50000\
epoch [4992/5000], loss:0.6461 Accuracy: 38655/50000\
epoch [4993/5000], loss:0.6461 Accuracy: 38589/50000\
epoch [4994/5000], loss:0.6466 Accuracy: 38591/50000\
epoch [4995/5000], loss:0.6470 Accuracy: 38631/50000\
epoch [4996/5000], loss:0.6466 Accuracy: 38609/50000\
epoch [4997/5000], loss:0.6466 Accuracy: 38590/50000\
epoch [4998/5000], loss:0.6455 Accuracy: 38606/50000\
epoch [4999/5000], loss:0.6474 Accuracy: 38618/50000\
epoch [5000/5000], loss:0.6463 Accuracy: 38612/50000"

print(sum([int(i.split("Accuracy: ")[-1]) for i in string.strip().split("/50000") if "Accuracy: " in i]) / 10)