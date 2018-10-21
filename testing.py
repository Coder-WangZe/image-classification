import os
import  tensorflow as tf
from utils import *


def test(model_id):
    # with tf.Graph().as_default() as gr:
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./new_motor_model/model.ckpt-' + str(model_id) + '.meta')
        gr = tf.get_default_graph()
        test_x = gr.get_tensor_by_name('input/x:0')
        test_y = gr.get_tensor_by_name('input/y:0')

        y_pred = gr.get_tensor_by_name('transfer_layer2/y_pred:0')
        # y_pred = tf.stop_gradient(y_pred)
        correct_pred = tf.equal(tf.argmax(y_pred, 1), test_y)
        test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        test_image, test_label = read_and_decode("motor_train_data.tfrecords")
        test_batch_image, test_batch_label = get_batch(test_image, test_label, batch_size=64)
    # with tf.Session(graph=gr) as sess:
        saver.restore(sess, './new_motor_model/model.ckpt-' + str(model_id))
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        batch_x, batch_y = sess.run([test_batch_image, test_batch_label])
        accuracy = sess.run(test_acc, feed_dict={test_x: batch_x, test_y: batch_y})
        print("****acc: ", accuracy)

        coord.request_stop()
        coord.join(threads)

test(118)
