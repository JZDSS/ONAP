import tensorflow as tf
from basenets import net
from tensorflow.contrib import slim
from basenets import resnet_v2


class Res50_CBAM(net.Net):

    def __init__(self, inputs, num_classes, name='ResNet', weight_decay=0.0004, **kwargs):
        super(Res50_CBAM, self).__init__(weight_decay=weight_decay, name=name, **kwargs)
        self.inputs = inputs
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
        self.num_classes = num_classes
        self.loss = None
        self.accuracy = None
        self.build()

    def build(self):
        with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=0.999)):
            logits, self.endpoints = resnet_v2.resnet_v2_50(self.inputs['images'],
                                                            num_classes=self.num_classes,
                                                            is_training=self.is_training)
        self.outputs['logits'] = tf.reshape(logits, [-1, self.num_classes])
        self.outputs['argmax'] = tf.argmax(self.outputs['logits'], axis=1, name='output/predict')

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
        pass


if __name__ == '__main__':

    x = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32)
    net = ResNet50({'images': x}, 5)
    pred = net.outputs['logits']

    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    # import cv2
    # img = cv2.imread('../images/dog.jpg')
    # img = cv2.resize(img, (227, 227))
    # img = img
    # img = np.expand_dims(img, 0)
    var_list = {v.op.name: v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES)}
    del var_list[u'resnet_v2_50/logits/biases']
    del var_list[u'resnet_v2_50/logits/weights']
    saver = tf.train.Saver(name='saver', var_list=var_list)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        saver.restore(sess, '../resnet_v2_50.ckpt_', )
        a = 1
        # print(sess.run(tf.argmax(pred, 0), feed_dict={x: img}))

