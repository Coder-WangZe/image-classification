import os
import tensorflow as tf
from utils import *
from time import time


def transfer_inference(input):
    with tf.name_scope(name="transfer_layer1"):
        weights = tf.get_variable(shape=[1024, 128], name="layer1_w")
        bias = tf.get_variable(shape=[128, ], name="layer1_b")
        fc_layer1 = tf.nn.relu(tf.add(tf.matmul(input, weights), bias))
    with tf.name_scope(name="transfer_layer2"):
        weights = tf.get_variable(shape=[128, 4], name="layer2_w")
        bias = tf.get_variable(shape=[4, ], name="layer2_b")
        fc_layer2 = tf.add(tf.matmul(fc_layer1, weights), bias, name="y_pred")
    return fc_layer2


def compute_loss(y_pred, y_true):
    labels = tf.one_hot(y_true, 4)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=y_pred))
    return loss


def train(lr):
    time1 = time()
    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph('./model/0.0001_64/model.ckpt-800.meta')
        train_x = g.get_tensor_by_name('input/x:0')
        train_y = g.get_tensor_by_name('input/y:0')

        layer_fc1 = g.get_tensor_by_name('inference/layer_fc1:0')
        feature_layer = tf.stop_gradient(layer_fc1)
        y_pred = transfer_inference(feature_layer)
        loss = compute_loss(y_pred, train_y)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), train_y)
        train_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        train_image, train_label = read_and_decode("motor_train_data.tfrecords")
        train_batch_image, train_batch_label = get_batch(train_image, train_label, batch_size=64)

        test_image, test_label = read_and_decode("motor_test.tfrecords")
        test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=64)
    with tf.Session(graph=g) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./model/0.0001_64/'))
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(500):
            batch_x, batch_y = sess.run([train_batch_image, train_batch_label])
            acc, loss_np, _ = sess.run([train_acc, loss, train_step], feed_dict={train_x: batch_x, train_y: batch_y})

            test_x, test_y = sess.run([test_batch_image, test_batch_label])
            test_acc = sess.run([train_acc], feed_dict={train_x: test_x, train_y: test_y})

            if i % 10 == 0:
                print('***************epochs:', i, '*************')
                print('***************train loss:', loss_np)
                print('***************train accruacy:', acc, '*************')
                print("***********test_accuracy:",  test_acc, "*********")
            if i > 100:
                saver.save(sess, './new_motor_model/model.ckpt', global_step=i+1)
        coord.request_stop()
        # queue需要关闭，否则报错
        coord.join(threads)
    time2 = round((time() - time1)/60, 2)
    print("cost time: ", time2)


train(lr=0.0001)