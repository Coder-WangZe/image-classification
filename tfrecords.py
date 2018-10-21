import tensorflow as tf
from PIL import Image
import os

current_path = "E:\\transfer_learning\\motor_data\\motor_fault_data\\"
writer = tf.python_io.TFRecordWriter("motor_train_data.tfrecords")

classes = ("fbo", "nor")

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

'''制作测试数据集'''

writer = tf.python_io.TFRecordWriter("motor_test.tfrecords")

classes = ("fbo_test", "nor_test")

for index, name in enumerate(classes):
   class_path = current_path + "/"+name + "/"
   for img_name in os.listdir(class_path):
       img_path = class_path + img_name
       img = Image.open(img_path)
       img = img.resize((240, 240))
       img_raw = img.tobytes()              #将图片转化为原生bytes
       example = tf.train.Example(features=tf.train.Features(feature={
           "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
           'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
       }))
       writer.write(example.SerializeToString())  #序列化为字符串


