import tensorflow as tf
from oi.panoreader_test import PANOReader
from basenets import resnet50


ckpt_dir = './ckpt'


def model_test(filename):
    images = PANOReader([filename], 1, 1, 1, num_epochs=1, drop_remainder=False, shuffle=False).read()
    inputs = {'images': images}
    net = resnet50.ResNet50(inputs, 5)
    prediction = tf.argmax(net.outputs['logits'], axis=1) + 1

    latest = tf.train.latest_checkpoint(ckpt_dir)
    saver = tf.train.Saver(name='saver')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    pre_list = []
    with tf.Session(config=config) as sess:

        saver.restore(sess, latest)
        while True:
            try:
                p = sess.run(prediction, feed_dict={net.is_training: False})
                pre_list.append(p[0])
                print p
            except tf.errors.OutOfRangeError as e:
                print 'done'
                break

    with open('result.txt', 'w') as f:
        for i in range(len(pre_list)):
            f.write("%d\n" % pre_list[i])

    return pre_list

if __name__ == '__main__':
    prediction = model_test('./data/TFcodeX_10.tfrecords')