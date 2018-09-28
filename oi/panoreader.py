import tensorflow as tf
import os
import cv2
import numpy as np
from oi import tfrecordsreader


class PANOReader(tfrecordsreader.TFRecordsReader):

    def __init__(self,
                 filenames,
                 batch_size,
                 num_readers,
                 read_threads,
                 num_epochs=None,
                 drop_remainder=True,
                 fake=False, **kwargs):
        super(PANOReader, self).__init__(filenames,
                                         batch_size,
                                         num_readers,
                                         read_threads,
                                         num_epochs,
                                         drop_remainder,
                                         fake, **kwargs)

    def _parser(self, record):
        tfrecord_features = tf.parse_single_example(record,
                                                    features={
                                                        'data': tf.FixedLenFeature([256, 256], dtype=tf.float32),
                                                        'label': tf.FixedLenFeature([], dtype=tf.int64)},
                                                    name='features')
        image = tfrecord_features['data']
        label = tfrecord_features['label']
        return image, label

    def _post_process(self, iterator):
        images, labels = iterator.get_next()
        labels = labels - 1
        images = (images + 1) * 128
        return images, labels

if __name__ == '__main__':
    data_dir = '../../data/PANO'
    filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.tfrecord'))
    images, labels = PANOReader(filenames, 1, 10, 4, num_epochs=1, drop_remainder=False).read()

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        n = 0
        while 1:
            n += 1
            try:
                i, l = sess.run([images, labels])
                print l
                # print os.path.join('../' + str(l[0]), str(n) + '.jpg')
                # cv2.imwrite(os.path.join('../' + str(l[0,]), str(n) + '.jpg'), i[0, :].astype(np.uint8))
            except Exception as e:
                print e
                break
