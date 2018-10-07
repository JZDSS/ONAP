import tensorflow as tf
import time
import os
from oi.panoreader import PANOReader
from basenets import alexnet
# tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
data_dir = './data'
ckpt_dir = './ckpt'

# with tf.device('/cpu:0'):
filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.tfrecords'))
images, labels = PANOReader(filenames, 32, 1, 4, num_epochs=None, drop_remainder=False).read()
inputs = {'images': images,
          'ground_truth': labels}
tf.summary.image('show', images, 1)
net = alexnet.AlexNet(inputs, 5)
net.calc_loss()


latest = tf.train.latest_checkpoint(ckpt_dir)
# saver = tf.train.import_meta_graph(latest + '.meta')
saver = tf.train.Saver(name='saver')
g = int(latest.split('-')[1]) if latest is not None else 0

summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/eval', sess.graph)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
    if g > 0:
        saver.restore(sess, latest)
        print('load ckpt %d' % g)

    curr = None
    while True:
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest != curr:
            g = int(latest.split('-')[1]) if latest is not None else 0
            saver.restore(sess, latest)

            summ = sess.run(summary_op)
            writer.add_summary(summ, g)
            writer.flush()
        curr = latest
        time.sleep(20)

    writer.close()