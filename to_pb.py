import tensorflow as tf
from basenets import resnet50
from tensorflow.python.framework.graph_util import convert_variables_to_constants

ckpt_dir = './ckpt'


def to_pb():
    images = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    inputs = {'images': images}
    net = resnet50.ResNet50(inputs, 5)

    latest = tf.train.latest_checkpoint(ckpt_dir)
    saver = tf.train.Saver(name='saver')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session() as sess:
        saver.restore(sess, latest)
        output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output/predict'])
        with tf.gfile.FastGFile('resnet50.pb', mode='wb') as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    prediction = to_pb()