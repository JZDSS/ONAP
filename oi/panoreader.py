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
                 shuffle=True,
                 fake=False, **kwargs):
        super(PANOReader, self).__init__(filenames,
                                         batch_size,
                                         num_readers,
                                         read_threads,
                                         num_epochs,
                                         drop_remainder,
                                         shuffle,
                                         fake, **kwargs)

    def _parser(self, record):
        tfrecord_features = tf.parse_single_example(record,
                                                    features={
                                                        'data': tf.FixedLenFeature([256, 256], dtype=tf.float32),
                                                        'label': tf.FixedLenFeature([], dtype=tf.int64)},
                                                    name='features')
        image = tfrecord_features['data']
        label = tfrecord_features['label']
        label = label - 1
        image = (image + 1) * 128
        image = tf.expand_dims(image, 2)

        image = tf.image.pad_to_bounding_box(image, 10, 10, 286, 286)
        image = tf.random_crop(image, [256, 256, 1])

        image = tf.concat([image, image, image], axis=2)
        return image, label

    def _post_process(self, iterator):
        images, labels = iterator.get_next()
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, 32)
        images = tf.image.random_hue(images, 0.05)
        images = tf.image.random_contrast(images, 0.5, 1.5)
        images = tf.image.random_saturation(images, 0.5, 1.5)

        # images = tf.image.per_image_standardization(images)
        return images, labels

if __name__ == '__main__':
    data_dir = '../data2/'
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
                cv2.imwrite(os.path.join('../images2/' + str(l[0,]), str(n) + '.jpg'), i[0, :].astype(np.uint8))
            except Exception as e:
                print e
                break
