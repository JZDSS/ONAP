import tensorflow as tf
import os
from oi.panoreader import PANOReader

data_dir = '../../data/PANO'
filenames = tf.train.match_filenames_once(os.path.join(data_dir, '*.tfrecord'))
images, labels = PANOReader(filenames, 1, 10, 4, num_epochs=1, drop_remainder=False).read()
