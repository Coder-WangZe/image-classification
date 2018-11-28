# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 00:54:22 2018

@author: Ze
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import tensorflow as tf
import os

def read_and_decode(filename):
    
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [240,240,3])
    
#    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#    img = tf.reshape(img, [14400])
    label = tf.cast(features['label'], tf.int64)

    return img, label

#image, label = read_and_decode("train.tfrecords")

#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(image, label, batch_size):
     
    images, label_batch = tf.train.shuffle_batch([image, label],batch_size=batch_size,
                                                 capacity=1000,min_after_dequeue=200)
    
    return images, tf.reshape(label_batch, [batch_size])

def get_test_batch(image, label, batch_size):
    
    images, label_batch = tf.train.shuffle_batch([image, label],batch_size=batch_size,
                                                 capacity=1000,min_after_dequeue=200)
    
    return images, tf.reshape(label_batch, [batch_size])


class network(object):
    def __init__(self):
        with tf.variable_scope("weights"):
            self.weights={
                
                'conv1':tf.get_variable('conv1',[5,5,3,16],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv2':tf.get_variable('conv2',[5,5,16,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv3':tf.get_variable('conv3',[4,4,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'conv4':tf.get_variable('conv4',[4,4,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                
                'fc1':tf.get_variable('fc1',[6*6*128,128],initializer=tf.contrib.layers.xavier_initializer()),
                
                'fc2':tf.get_variable('fc2',[128,10],initializer=tf.contrib.layers.xavier_initializer()),
                }
        with tf.variable_scope("biases"):
            self.biases={
                'conv1':tf.get_variable('conv1',[16,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2':tf.get_variable('conv2',[32,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3':tf.get_variable('conv3',[64,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv4':tf.get_variable('conv4',[128,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                
                'fc1':tf.get_variable('fc1',[128,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2':tf.get_variable('fc2',[10,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

    def inference(self,images):
        
        images = tf.reshape(images, shape=[-1, 240,240, 3])
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理

        #第一层:120*120*3  >>>  100*100*20  >>>  50*50*20
        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv1'])
        relu1= tf.nn.relu(conv1)
        
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        
        #第二层: 50*50*20  >>>  48*48*40    >>>  24*24*40
        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv2'])
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 第三层: 24*24*40  >>>  20*20*60  >>>  10*10*60
        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv3'])
        relu3= tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
         
        # 第四层：10*10*60  >>> 5*5*80
        conv4=tf.nn.bias_add(tf.nn.conv2d(pool3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv4'])
        
        pool4=tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool5=tf.nn.max_pool(pool4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool5, [-1, self.weights['fc1'].get_shape().as_list()[0]])
        # dropout 正则化
        drop1=tf.nn.dropout(flatten,0.5)
        
        fc1=tf.matmul(drop1, self.weights['fc1'])+self.biases['fc1']

        fc_relu1=tf.nn.relu(fc1)

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']

        return  fc2
        
    def inference_test(self,images):
        
        images = tf.reshape(images, shape=[-1, 240,240, 3])
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理

        #第一层:120*120*3  >>>  100*100*20  >>>  50*50*20
        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv1'])
        relu1= tf.nn.relu(conv1)
        
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        #第二层: 50*50*20  >>>  48*48*40    >>>  24*24*40
        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv2'])
        relu2= tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        # 第三层: 24*24*40  >>>  20*20*60  >>>  10*10*60
        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv3'])
        relu3= tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
         
        # 第四层：10*10*60  >>> 5*5*80
        conv4=tf.nn.bias_add(tf.nn.conv2d(pool3, self.weights['conv4'], strides=[1, 1, 1, 1], padding='VALID'),
                             self.biases['conv4'])
        
        pool4=tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool5=tf.nn.max_pool(pool4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool5, [-1, self.weights['fc1'].get_shape().as_list()[0]])
        # dropout 正则化
        
        
        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1']

        fc_relu1=tf.nn.relu(fc1)

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']

        return  fc2

    #计算softmax交叉熵损失函数
    def sorfmax_loss(self,predicts,labels):
        
        labels=tf.one_hot(labels,self.weights['fc2'].get_shape().as_list()[1])  #as——list得到第二个维度（2分类为2）
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicts, labels=labels))
        self.cost= loss
        return self.cost
    #梯度下降
    def optimer(self,loss,learning_rate):
        train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return train_optimizer


def train(learning_rate,num_epochs,batch_size):
    tf.reset_default_graph()

    #获取训练集
    image, label = read_and_decode("train_data.tfrecords")
    batch_image,batch_label=get_batch(image, label, batch_size=batch_size)
    #获取测试集
    test_image,test_label=read_and_decode("test_data.tfrecords")
    test_images,test_labels=get_test_batch(test_image,test_label,batch_size=100)#batch 生成测试
    
    #计算训练集结果并优化
    net=network()
    inf=net.inference(batch_image) #正向传播
    loss=net.sorfmax_loss(inf,batch_label)
    opti=net.optimer(loss,learning_rate)
    time1=time.time()
    test_inf=net.inference_test(test_images)
    #训练集准确率
    correct_prediction = tf.equal(tf.cast(tf.argmax(inf,1),tf.int64), batch_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #测试集准确率
    correct_test_prediction=tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int64), test_labels)
    accuracy_test = tf.reduce_mean(tf.cast(correct_test_prediction, tf.float32))
    saver=tf.train.Saver()
    costs=[]
    accuracy_tests=[]
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for i in range(num_epochs):
            loss_np,_,label_np,image_np,inf_np=session.run([loss,opti,batch_label,batch_image,inf])
            costs.append(loss_np)
            accuracy_np=session.run([accuracy])
            accuracy__test=session.run(accuracy_test)
            accuracy_tests.append(accuracy__test)
            
            if i % 20 == 0:
                print( 'trainloss:',loss_np)
                print( '***************train accruacy:',accuracy_np,'*************')
                print('***************test accruacy:',accuracy__test,'************')
                print('***************循环次数i:',i,'/',num_epochs,'**************')
            if i > 199 and i % 2 == 0:
                saver.save(session, 'model'+"\\" + '/model.ckpt', global_step=i+1)
 
        coord.request_stop() #queue需要关闭，否则报错
        coord.join(threads)
        time2=time.time()
        time3=(time2-time1)/60
        time3=round(time3,2)
        print("耗时：", time3, "min")
        return costs, accuracy_tests, time3

'''batch不变，不同学习率绘图'''

train(learning_rate=0.001, num_epochs=500, batch_size=32)








