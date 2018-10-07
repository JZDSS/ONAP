import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from basenets import net
import numpy as np


class ResNet(net.Net):

    def __init__(self, inputs, num_classes, name='ResNet', npy_path=None, weight_decay=0.0004, **kwargs):
        super(ResNet, self).__init__(weight_decay=weight_decay, name=name, **kwargs)
        self.inputs = inputs
        self.npy_path = npy_path
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
        self.num_classes = num_classes
        self.loss = None
        self.accuracy = None
        self.build()
        if self.npy_path:
            self.setup()

    def set_npy_path(self, path):
        self.npy_path = path

    def build(self):
        def _block(self, inputs, num_outputs, weight_decay, scope, is_training, down_sample=False):
            with tf.variable_scope(scope):
                num_inputs = inputs.get_shape().as_list()[3]

                res = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3],
                                    strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                                    scope='conv1', b_norm=True, is_training=is_training,
                                    weight_decay=weight_decay)

                res = layers.conv2d(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                                    scope='conv2', b_norm=True, is_training=is_training,
                                    weight_decay=weight_decay)
                if num_inputs != num_outputs:
                    inputs = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[1, 1],
                                           activation_fn=None,
                                           scope='brantch_a', strides=[1, 2, 2, 1], b_norm=True,
                                           is_training=is_training,
                                           weight_decay=weight_decay)
                res = tf.nn.relu(res + inputs)
            self.architecture.append(scope + '\n')
            return res
        endpoints = self.endpoints
        y = self.inputs['images']
        with arg_scope([layers.conv2d], activation_fn=tf.nn.relu,
                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                       biases_regularizer=layers.l2_regularizer(self.weight_decay)):
            y = layers.conv2d(y, 96, (11, 11), 4, 'VALID', scope='conv1')
            endpoints['conv1'] = y
            y = tf.nn.lrn(y, 5, 1, 0.0001, 0.75)
            y = layers.max_pool2d(y, (3, 3), 2, 'VALID', scope='pool1')
            y1, y2 = tf.split(y, 2, 3)
            y1 = layers.conv2d(y1, 128, (5, 5), 1, 'SAME', scope='conv2_1')
            y2 = layers.conv2d(y2, 128, (5, 5), 1, 'SAME', scope='conv2_2')
            endpoints['conv2_1'] = y1
            endpoints['conv2_2'] = y2
            y = tf.concat((y1, y2), 3)
            endpoints['conv2'] = y
            y = tf.nn.lrn(y, 5, 1, 0.0001, 0.75)
            y = layers.max_pool2d(y, (3, 3), 2, 'VALID', scope='pool2')
            y = layers.conv2d(y, 384, (3, 3), 1, 'SAME', scope='conv3')
            endpoints['conv3'] = y
            y1, y2 = tf.split(y, 2, 3)
            y1 = layers.conv2d(y1, 192, (3, 3), 1, 'SAME', scope='conv4_1')
            y2 = layers.conv2d(y2, 192, (3, 3), 1, 'SAME', scope='conv4_2')
            endpoints['conv4_1'] = y1
            endpoints['conv4_2'] = y2
            y1 = layers.conv2d(y1, 128, (3, 3), 1, 'SAME', scope='conv5_1')
            y2 = layers.conv2d(y2, 128, (3, 3), 1, 'SAME', scope='conv5_2')
            endpoints['conv5_1'] = y1
            endpoints['conv5_2'] = y2
            y = tf.concat([y1, y2], 3)
            endpoints['conv5'] = y
            y = layers.max_pool2d(y, (3, 3), 2, 'VALID', scope='pool5')
            y = layers.conv2d(y, 4096, (6, 6), 1, 'VALID', scope='fc6')
            endpoints['fc6'] = y
            y = layers.conv2d(y, 4096, (1, 1), 1, 'VALID', scope='fc7')
            endpoints['fc7'] = y
            y = layers.conv2d(y, self.num_classes, (1, 1), 1, 'VALID', scope='fc8', activation_fn=None)
            endpoints['fc8'] = y
            self.outputs['logits'] = tf.reshape(y, [-1, self.num_classes])

    def calc_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.inputs['ground_truth'],
                                                               self.outputs['logits'])

        with tf.name_scope('accuracy'):
            # a = tf.reshape(tf.argmax(self.outputs['logits'], 1), [-1, 1])
            correct_prediction = tf.equal(
                tf.reshape(tf.argmax(self.outputs['logits'], 1), [-1, 1]),
                tf.reshape(self.inputs['ground_truth'], [-1, 1]))
            # a = tf.to_float(correct_prediction)
            self.accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def get_update_ops(self):
        return []

    def setup(self):
        """Define ops that load pre-trained vgg16 net's weights and biases and add them to tf.GraphKeys.INIT_OP
        collection.
        """

        # caffe-tensorflow/convert.py can only run with Python2. Since the default encoding format of Python2 is ASCII
        # but the default encoding format of Python3 is UTF-8, it will raise an error without 'encoding="latin1"'
        weight_dict = np.load(self.npy_path, encoding="latin1").item()
        scopes = ['conv1', 'conv3']
        for scope in scopes:
            with tf.variable_scope(scope, reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(weight_dict[scope]['weights'])
                b_init_op = biases.assign(weight_dict[scope]['biases'])
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        with tf.variable_scope('fc6', reuse=True):
            weights = tf.get_variable('weights')
            biases = tf.get_variable('biases')
            w = weight_dict['fc6']['weights']
            w = np.reshape(w, (6, 6, 256, 4096))
            w_init_op = weights.assign(w)
            b_init_op = biases.assign(weight_dict['fc6']['biases'])
            tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        scopes = ['fc7', 'fc8'] if self.num_classes == 1000 else ['fc7']
        for scope in scopes:
            with tf.variable_scope(scope, reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w = weight_dict[scope]['weights']
                w = np.expand_dims(w, 0)
                w = np.expand_dims(w, 0)
                w_init_op = weights.assign(w)
                b_init_op = biases.assign(weight_dict[scope]['biases'])
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        scopes = ['conv2', 'conv4', 'conv5']
        for scope in scopes:
            w = weight_dict[scope]['weights']
            w1, w2 = np.split(w, 2, 3)
            b = weight_dict[scope]['biases']
            b1, b2 = np.split(b, 2, 0)
            with tf.variable_scope(scope + '_1', reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(w1)
                b_init_op = biases.assign(b1)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)
            with tf.variable_scope(scope + '_2', reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(w2)
                b_init_op = biases.assign(b2)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)


if __name__ == '__main__':

    x = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32)
    net = AlexNet(x, npy_path='../npy/alexnet.npy')
    pred = net.outputs['logits']

    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    import cv2
    img = cv2.imread('../images/dog.jpg')
    img = cv2.resize(img, (227, 227))
    img = img
    img = np.expand_dims(img, 0)
    with tf.Session() as sess:
        sess.run(init_ops)
        print(sess.run(tf.argmax(pred, 0), feed_dict={x: img}))

