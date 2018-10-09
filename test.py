import tensorflow as tf
from oi.panoreader_test import PANOReader


def model_test(filename):
    images = PANOReader([filename], 1, 1, 1, num_epochs=1, drop_remainder=False, shuffle=False).read()
    x = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    is_training = tf.placeholder(shape=[], dtype=tf.bool)
    with open('./resnet50.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output = tf.import_graph_def(graph_def,
                                     input_map={'Placeholder:0': x,
                                                'Placeholder_1:0': is_training},
                                     return_elements=['output/predict:0'])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    pre_list = []
    with tf.Session(config=config) as sess:

        while True:
            try:
                i = sess.run(images)
                p = sess.run(output[0] + 1, feed_dict={x: i, is_training: False})
                pre_list.append(p[0])
                print p
            except tf.errors.OutOfRangeError as e:
                print 'finished or can not find file!'
                break

    with open('result.txt', 'w') as f:
        for i in range(len(pre_list)):
            f.write("%d\n" % pre_list[i])

    return pre_list

if __name__ == '__main__':
    prediction = model_test('./data/TFcodeX_10.tfrecords')