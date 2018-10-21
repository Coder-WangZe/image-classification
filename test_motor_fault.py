import os
import  tensorflow as tf
from utils import *


def tt(a):
    with tf.Graph().as_default() as g:
        saver = tf.train.import_meta_graph('./new_motor_model/model.ckpt-500.meta')
        test_x = g.get_tensor_by_name('input/x:0')
        test_y = g.get_tensor_by_name('input/y:0')

        y_pred = g.get_tensor_by_name('transfer_layer2/y_pred:0')
        correct_pred = tf.equal(tf.argmax(y_pred, 1), test_y)
        test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        test_image, test_label = read_and_decode("motor_test_data.tfrecords")
        test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=64)
    with tf.Session(graph=g) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./new_motor_model/'))
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        x, y = sess.run([test_batch_image, test_batch_label])
        accuracy = sess.run([test_acc], feed_dict={test_x: x, test_y: y})
        print("****acc: ", accuracy)

        coord.request_stop()
        coord.join(threads)

tt(a=1)
n=2