import functools

import tensorflow as tf

# Module structure based off of
# https://danijar.com/structuring-your-tensorflow-models/


def lazy_property(function):

    attribute = '__cache__' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class BiLSTM(object):

    def __init__(self, data, target_A, target_B,
                 vocab_size, embedding_size=100, lstm_size=100,
                 learning_rate=0.1):
        self.data = data
        self.target_A = target_A
        self.target_B = target_B

        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size

        self._learning_rate = learning_rate

        self.embedded_data
        self.lstm_output

        self.prediction_A
        self.optimize_A
        self.error_A

        self.prediction_B
        self.optimize_B
        self.error_B

    @lazy_property
    def length(self):
        used = tf.sign(tf.abs(self.data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def mask_A(self):
        max_length = tf.shape(self.target_A)[1]
        return tf.sequence_mask(self.length, maxlen=max_length,
                                dtype=tf.float32)

    @lazy_property
    def mask_B(self):
        max_length = tf.shape(self.target_B)[1]
        return tf.sequence_mask(self.length, maxlen=max_length,
                                dtype=tf.float32)

    @lazy_property
    def embedded_data(self):
        # Word embeddings
        embedding_matrix = self._random_variable(
            [self._vocab_size, self._embedding_size], uniform=True)
        return tf.nn.embedding_lookup(embedding_matrix, self.data)

    @lazy_property
    def lstm_output(self):
        # Bidirectional RNN
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(self._lstm_size),
            tf.nn.rnn_cell.LSTMCell(self._lstm_size),
            self.embedded_data, sequence_length=self.length, dtype=tf.float32)

        fw_output, bw_output = output
        return tf.concat([fw_output, bw_output], 2)

    @lazy_property
    def prediction_A(self):
        # Softmax layer
        max_length = tf.shape(self.target_A)[1]
        num_classes = int(self.target_A.get_shape()[2])

        weight = self._random_variable([self._lstm_size * 2, num_classes])
        bias = self._random_variable([num_classes])

        output = tf.reshape(self.lstm_output, [-1, self._lstm_size * 2])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])

        return prediction

    @lazy_property
    def prediction_B(self):
        # Softmax layer
        max_length = tf.shape(self.target_B)[1]
        num_classes = int(self.target_B.get_shape()[2])

        weight = self._random_variable([self._lstm_size * 2, num_classes])
        bias = self._random_variable([num_classes])

        output = tf.reshape(self.lstm_output, [-1, self._lstm_size * 2])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])

        return prediction

    @lazy_property
    def cost_A(self):
        # Compute cross entropy for each frame
        cross_entropy = self.target_A * tf.log(self.prediction_A)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)

        cross_entropy *= self.mask_A

        # Average over actual sequence lengths
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def cost_B(self):
        # Compute cross entropy for each frame
        cross_entropy = self.target_B * tf.log(self.prediction_B)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)

        cross_entropy *= self.mask_B

        # Average over actual sequence lengths
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize_A(self):
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        return optimizer.minimize(self.cost_A)

    @lazy_property
    def optimize_B(self):
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        return optimizer.minimize(self.cost_B)

    @lazy_property
    def error_A(self):
        mistakes = tf.not_equal(tf.argmax(self.target_A, 2),
                                tf.argmax(self.prediction_A, 2))
        mistakes = tf.cast(mistakes, tf.float32)

        mistakes *= self.mask_A

        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @lazy_property
    def error_B(self):
        mistakes = tf.not_equal(tf.argmax(self.target_B, 2),
                                tf.argmax(self.prediction_B, 2))
        mistakes = tf.cast(mistakes, tf.float32)

        mistakes *= self.mask_B

        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @staticmethod
    def _random_variable(shape, uniform=False):
        if uniform:
            return tf.Variable(tf.random_uniform(shape, -1.0, 1.0))
        else:
            return tf.Variable(tf.random_normal(shape))
