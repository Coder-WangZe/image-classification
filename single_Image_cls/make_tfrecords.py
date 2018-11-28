# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:22:46 2018

@author: Ze
"""

import os
os.chdir("D:\\project2\\cls\\")
import tensorflow as tf
from PIL import Image

current_path="D:\\project2\\cls\\train1"

writer = tf.python_io.TFRecordWriter("train_data.tfrecords")

classes = ("smoke", "no_smoke")

for index, name in enumerate(classes):
    class_path = current_path + "/"+name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((240,240))
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()


current_path="D:\\project2\\cls\\test"
writer = tf.python_io.TFRecordWriter("test_data.tfrecords")

classes=("smoke","no_smoke")

for index, name in enumerate(classes):
    class_path = current_path + "/"+name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((240,240))
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串 
writer.close()














