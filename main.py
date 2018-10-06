import tensorflow as tf
import os
from oi.panoreader import PANOReader
from basenets import alexnet

data_dir = './data'
filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.tfrecord'))
images, labels = PANOReader(filenames, 10, 9, 4, num_epochs=None, drop_remainder=False).read()
inputs = {'images': images,
          'ground_truth': labels}
net = alexnet.AlexNet(inputs, 5, npy_path='./npy/alexnet.npy')
net.calc_loss()

opt = tf.train.MomentumOptimizer(0.001, 0.9)
train_op = opt.minimize(net.loss)

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))

    for i in range(10000):
        sess.run([train_op])
        if i % 100 == 0:
            print(sess.run([net.loss, net.accuracy]))