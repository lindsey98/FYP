import os

for i in range(2, 134):
    os.rename("./data/D_S/" + str(i) + "/cropped.png", "./data/D_S/" + str(i) + "/yolo_box.png")
    os.rename("./data/D_S/" + str(i) + "/pred_target.png", "./data/D_S/" + str(i) + "/logo.png")
