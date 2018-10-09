import tensorflow as tf
import os
from oi.panoreader import PANOReader
from basenets import resnet50
from basenets import res50_CBAM
# tf.enable_eager_execution()

data_dir = './data2'
ckpt_dir = './ckpt'
batch_size = 8

filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.tfrecord'))
images, labels = PANOReader(filenames, batch_size, 9, 4, num_epochs=None, drop_remainder=False, shuffle=True).read()
inputs = {'images': images,
          'ground_truth': labels}
tf.summary.image('show', images, 1)
net = res50_CBAM.Res50_CBAM(inputs, 5)
net.calc_loss()
var_list = {v.op.name: v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES)}
del var_list[u'resnet_v2_50/logits/biases']
del var_list[u'resnet_v2_50/logits/weights']
for key in var_list.keys():
    if key.find('CBAM') != -1:
        del var_list[key]
saver0 = tf.train.Saver(name='saver0', var_list=var_list)

saver = tf.train.Saver(max_to_keep=5,
                       keep_checkpoint_every_n_hours=2,
                       name='saver')
latest = tf.train.latest_checkpoint(ckpt_dir)

g = int(latest.split('-')[1]) if latest is not None else 0
global_step = tf.Variable(g, name='global_step', trainable=False)
learning_rate = tf.train.piecewise_constant(global_step, [50000, 75000], [0.001, 0.0001, 0.00001])
# learning_rate = tf.train.exponential_decay(0.01, global_step, 200, 0.99)
tf.summary.scalar('lr', learning_rate)
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = opt.minimize(net.loss, global_step=global_step)

summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter('logs/train', sess.graph)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    saver0.restore(sess, 'resnet_v2_50.ckpt', ) # download from http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
    sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
    if g > 0:
        # saver.recover_last_checkpoints(ckpt_dir)
        saver.restore(sess, latest)
        print('load ckpt %d' % g)

    for i in range(g, 100000):
        sess.run(train_op, feed_dict={net.is_training: True})
        if i % 100 == 0:
            summ = sess.run(summary_op, feed_dict={net.is_training: False})
            writer.add_summary(summ, i)
            writer.flush()
            saver.save(sess, os.path.join(ckpt_dir, net.name), i)
    writer.close()