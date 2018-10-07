import six
from abc import ABCMeta, abstractmethod
from oi.reader import Reader
import tensorflow as tf


@six.add_metaclass(ABCMeta)
class TFRecordsReader(Reader):

    def __init__(self,
                 filenames,
                 batch_size,
                 num_readers,
                 read_threads,
                 num_epochs=None,
                 drop_remainder=True,
                 fake=False, **kwargs):
        super(TFRecordsReader, self).__init__(batch_size, fake, **kwargs)
        self.filenames = filenames
        self.num_readers = num_readers
        self.read_threads = read_threads
        self.num_epochs = num_epochs
        self.drop_remainder = drop_remainder

    @abstractmethod
    def _parser(self, record):
        raise NotImplementedError

    def _get_iterator(self, filenames, num_readers, read_threads, num_epochs):
        # TensorFlow 1.6
        dataset = tf.data.Dataset.list_files(filenames)
        # higher version TensorFlow
        # dataset = tf.data.Dataset.list_files(filenames, True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset, num_readers, 1)
        dataset = dataset.map(self._parser, read_threads)
        dataset = dataset.shuffle(buffer_size=3 * self.batch_size)
        # TensorFlow version 1.6
        if self.drop_remainder:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        else:
            dataset = dataset.batch(self.batch_size)
        # Tensorflow version 1.10+
        # dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.repeat(num_epochs)
        try:
            iterator = dataset.make_one_shot_iterator()
        except ValueError:
            iterator = dataset.make_initializable_iterator()
            tf.add_to_collection(tf.GraphKeys.INIT_OP, iterator.initializer)
        return iterator

    def read(self):
        iterator = None
        if not self.fake:
            iterator = self._get_iterator(self.filenames,
                                          self.num_readers,
                                          self.read_threads,
                                          self.num_epochs)
        return self._post_process(iterator)

    def _post_process(self, iterator):
        return iterator.get_next()
