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
                                                        'data': tf.FixedLenFeature([256, 256], dtype=tf.float32)},
                                                    name='features')
        image = tfrecord_features['data']
        image = (image + 1) * 128
        image = tf.expand_dims(image, 2)
        image = tf.concat([image, image, image], axis=2)
        return image

    def _post_process(self, iterator):
        images = iterator.get_next()

        return images
