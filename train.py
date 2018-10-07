import tensorflow as tf
import os
from oi.panoreader import PANOReader
from basenets import alexnet
# tf.enable_eager_execution()

data_dir = './data'
ckpt_dir = './ckpt'

filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.tfrecord'))
images, labels = PANOReader(filenames, 32, 9, 4, num_epochs=None, drop_remainder=False).read()
inputs = {'images': images,
          'ground_truth': labels}
tf.summary.image('show', images, 1)
net = alexnet.AlexNet(inputs, 5, npy_path='./npy/alexnet.npy')
net.calc_loss()

saver = tf.train.Saver(max_to_keep=5,
                       keep_checkpoint_every_n_hours=2,
                       name='saver')
latest = tf.train.latest_checkpoint(ckpt_dir)

g = int(latest.split('-')[1]) if latest is not None else 0
global_step = tf.Variable(g, name='global_step', trainable=False)
# learning_rate = tf.train.piecewise_constant(global_step, [2000, 4000], [0.001, 0.0001, 0.00001])
learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.99)
tf.summary.scalar('lr', learning_rate)
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
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
    sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
    if g > 0:
        # saver.recover_last_checkpoints(ckpt_dir)
        saver.restore(sess, os.path.join(ckpt_dir, latest))
        print('load ckpt %d' % g)

    for i in range(g, 10000):
        sess.run([train_op])
        if i % 100 == 0:
            summ = sess.run(summary_op)
            writer.add_summary(summ, i)
            writer.flush()
            saver.save(sess, os.path.join(ckpt_dir, net.name), i)
    writer.close()