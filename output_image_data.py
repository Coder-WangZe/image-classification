import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl


def normalize_data(data1):

    data_max = np.max(data1)
    data_min = np.min(data1)
    norm_data = 2 * (data1 - data_min) / (data_max - data_min) -1
    return norm_data


def draw(raw_data_path, save_path, fault_name, out_file_name, T=1200, repeat_rate=2 / 3, linewidth=0.7):
    raw_data = np.loadtxt(raw_data_path)
    # default draw argument
    mpl.rcParams['figure.figsize'] = (4, 3)
    number = int(len(raw_data) // (T * (1 - repeat_rate))) - 1
    start_signal = int(T * (1 - repeat_rate))
    x_min, x_max = 0, 1200
    y_min, y_max = -1, 1
    # norm the data
    data = normalize_data(raw_data)
    for index in range(number):
        # 每一段信号再归一化到（-1,1），三分之一重复率，每1200点截取一张图保存
        fig = plt.figure()
        ax = plt.axes()
        x = data[index * start_signal:index * start_signal + T]
        x = normalize_data(x)
        ax.plot(x, linewidth=linewidth, color="b")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        fig.savefig(out_file_name + str(index) + ".jpg", dpi=100)
        plt.close()


motor_data_path = "E:\\transfer_learning\\motor_data\\Fault simulator(txt)\\"
fault_class = ['amis', 'br', 'brb', 'fbo', 'mun', 'nor', 'pmis', 'pun1']

for i in range(8):
    fault_name = fault_class[i]
    save_path = "E:\\transfer_learning\\motor_data\\time_domain_pic\\" + fault_name + "\\"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for j in range(1, 21):
        if j < 10:
            data_path = motor_data_path + fault_name + "_0" + str(j) + "_" + "a1.txt"
        else:
            data_path = motor_data_path + fault_name + "_" + str(j) + "_" + "a1.txt"
        out_file_name = save_path + fault_name + "_" + str(j) + "_"
        draw(data_path, save_path, fault_name, out_file_name)


