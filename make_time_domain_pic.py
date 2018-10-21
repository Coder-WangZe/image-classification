# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:52:58 2018

@author: Ze
"""
rotate_speed = "1797转速"  # 转速
degree = "14故障程度"  # 故障类型（ball/inner/outer_damage）

import time
import os

os.chdir("E:\\故障4分类\\时域图数据\\" + rotate_speed + "\\" + degree + "\\")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shutil
from PIL import Image
import tensorflow as tf

time1 = time.time()
# 加载文件
normal_data = np.loadtxt("N.txt")
inner_data = np.loadtxt("I.txt")
ball_data = np.loadtxt("B.txt")
outer_data = np.loadtxt("O.txt")
# 转为一维矩阵
normal_data = np.reshape(normal_data, (normal_data.shape[0], 1))
inner_data = np.reshape(inner_data, (inner_data.shape[0], 1))
ball_data = np.reshape(ball_data, (ball_data.shape[0], 1))
outer_data = np.reshape(outer_data, (outer_data.shape[0], 1))


# 定义归一化函数
def Normalize(data, style):
    max = np.max(data)
    min = np.min(data)
    # 选择归一化到（0,1）还是（-1,1）
    if style == (0, 1):
        Normalized_data = (data - min) / (max - min)
    elif style == (-1, 1):
        Normalized_data = 2 * (data - min) / (max - min) - 1
    return Normalized_data


'''设置图像大小,出图量，取样周期，重复率，线条宽度，X轴最值，Y轴最值'''
mpl.rcParams['figure.figsize'] = (4, 3)
number = 300
T = 1200
repeat_rate = 2 / 3
start_signal = int(T * (1 - repeat_rate))
linewidth = 0.7
x_min, x_max = 0, 1200
y_min, y_max = -1, 1
# 创建相应的文件夹存放不同程度的故障数据
current_path1 = os.getcwd()
os.makedirs(current_path1 + "\\data\\normal_data")
os.makedirs(current_path1 + "\\data\\inner_data")
os.makedirs(current_path1 + "\\data\\ball_data")
os.makedirs(current_path1 + "\\data\\outer_data")
# 保存路径
save_path = current_path1 + "\\data"

# 出图并保存
for i in range(number):
    # 每一段信号再归一化到（-1,1），三分之一重复率，每1200点截取一张图保存
    plt.figure()
    x = normal_data[i * start_signal:i * start_signal + T, :]
    x = Normalize(x, (-1, 1))
    plt.plot(x, linewidth=linewidth, color="b")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(save_path + "\\normal_data\\" + degree + "normal_" + str(i) + ".jpg", dpi=100)
    plt.close()

for i in range(number):
    # 每一段信号再归一化到（-1,1），三分之一重复率，每1200点截取一张图保存
    plt.figure()
    x = inner_data[i * start_signal:i * start_signal + T, :]
    x = Normalize(x, (-1, 1))
    plt.plot(x, linewidth=linewidth, color="b")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(save_path + "\\inner_data\\" + degree + "inner_" + str(i) + ".jpg", dpi=100)
    plt.close()

for i in range(number):
    # 每一段信号再归一化到（-1,1），三分之一重复率，每1200点截取一张图保存
    plt.figure()
    x = ball_data[i * start_signal:i * start_signal + T, :]
    x = Normalize(x, (-1, 1))
    plt.plot(x, linewidth=linewidth, color="b")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(save_path + "\\ball_data\\" + degree + "ball_" + str(i) + ".jpg", dpi=100)
    plt.close()

for i in range(number):
    # 每一段信号再归一化到（-1,1），三分之一重复率，每1200点截取一张图保存
    plt.figure()
    x = outer_data[i * start_signal:i * start_signal + T, :]
    x = Normalize(x, (-1, 1))
    plt.plot(x, linewidth=linewidth, color="b")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(save_path + "\\outer_data\\" + degree + "outer_" + str(i) + ".jpg", dpi=100)
    plt.close()

'''分离训练集和验证集'''
current_path = os.getcwd() + "\\data" + "\\"
sub_dir_1 = os.listdir(current_path)

for i in range(len(sub_dir_1)):
    sub_dir_2 = os.listdir(current_path + "\\" + sub_dir_1[i])
    dir_path = current_path + "\\" + sub_dir_1[i] + "\\"

    os.makedirs(current_path + sub_dir_1[i] + "_test")
    des_path = current_path + "\\" + sub_dir_1[i] + "_test"
    for m in range(number):
        dir_name = dir_path + sub_dir_2[m]
        random = np.random.randint(100)
        if random < 20:
            shutil.move(dir_name, des_path)

'''制作训练数据集'''
# writer = tf.python_io.TFRecordWriter("data\\train_data"+rotate_speed+"_"+degree+".tfrecords")
#
# classes=(("normal_data","inner_data","ball_data","outer_data"))
#
# for index, name in enumerate(classes):
#    class_path = current_path + "/"+name + "/"
#    for img_name in os.listdir(class_path):
#        img_path = class_path + img_name
#        img = Image.open(img_path)
#        img = img.resize((240,240))
#        img_raw = img.tobytes()              #将图片转化为原生bytes
#        example = tf.train.Example(features=tf.train.Features(feature={
#            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#        }))
#        writer.write(example.SerializeToString())  #序列化为字符串
# writer.close()
#
# '''制作测试数据集'''
#
# writer = tf.python_io.TFRecordWriter("data\\test_data"+rotate_speed+"_"+degree+".tfrecords")
#
# classes=(("normal_data_test","inner_data_test","ball_data_test","outer_data_test"))
#
# for index, name in enumerate(classes):
#    class_path = current_path + "/"+name + "/"
#    for img_name in os.listdir(class_path):
#        img_path = class_path + img_name
#        img = Image.open(img_path)
#        img = img.resize((240,240))
#        img_raw = img.tobytes()              #将图片转化为原生bytes
#        example = tf.train.Example(features=tf.train.Features(feature={
#            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#        }))
#        writer.write(example.SerializeToString())  #序列化为字符串
#
#
#
time2 = time.time()
time3 = round((time2 - time1) / 60, 2)
print("制作数据集耗时：", time3, "min")














