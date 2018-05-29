import functools

import tensorflow as tf
import tf_utils

import crf

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

    def __init__(self, num_classes, vocab_size, shape_domain_size,
                 char_domain_size, char_size, embedding_size, shape_size,
                 nonlinearity, viterbi, hidden_dim, char_embeddings,
                 embeddings=None):

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.shape_domain_size = shape_domain_size
        self.char_domain_size = char_domain_size
        self.char_size = char_size
        self.embedding_size = embedding_size
        self.shape_size = shape_size
        self.nonlinearity = nonlinearity
        self.viterbi = viterbi
        self.hidden_dim = hidden_dim
        self.char_embeddings = char_embeddings
        self.embeddings = embeddings

        # word embedding input
        self.input_x1 = tf.placeholder(tf.int64, [None, None], name="input_x1")

        # shape embedding input
        self.input_x2 = tf.placeholder(tf.int64, [None, None], name="input_x2")

        # labels
        self.input_y = tf.placeholder(tf.int64, [None, None], name="input_y")
        self.label_mask = tf.placeholder(tf.float32, [None, None, num_classes], name="label_mask")

        # padding mask
        self.input_mask = tf.placeholder(tf.float32, [None, None], name="input_mask")

        self.batch_size = tf.placeholder(tf.int32, None, name="batch_size")

        self.max_seq_len = tf.placeholder(tf.int32, None, name="max_seq_len")

        # sequence lengths
        self.sequence_lengths = tf.placeholder(tf.int32, [None, None], name="sequence_lengths")

        # dropout and l2 penalties
        self.middle_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="middle_dropout_keep_prob")
        self.hidden_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="hidden_dropout_keep_prob")
        self.input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        self.word_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="word_dropout_keep_prob")

        self.l2_penalty = tf.placeholder_with_default(0.0, [], name="l2_penalty")

        self.projection = tf.placeholder_with_default(False, [], name="projection")

        self.drop_penalty = tf.placeholder_with_default(0.0, [], name="drop_penalty")

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        # set the pad token to a constant 0 vector
        self.word_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, embedding_size])
        self.shape_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, shape_size])
        self.char_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, char_size])

        self.use_characters = char_size != 0
        self.use_shape = shape_size != 0

        if self.viterbi:
            self.transition_params = tf.get_variable("transitions", [num_classes, num_classes])

        word_embeddings_shape = (vocab_size - 1, embedding_size)
        self.w_e = tf_utils.initialize_embeddings(word_embeddings_shape, name="w_e", pretrained=embeddings)

        self.flat_sequence_lengths
        self.lstm_output
        self.lstm_output_no_drop
        self.unflat_scores
        self.unflat_no_dropout_scores
        self.predictions
        self.loss
        self.test

    @lazy_property
    def flat_sequence_lengths(self):
        nonzero_elements = tf.not_equal(self.sequence_lengths, tf.zeros_like(self.sequence_lengths))
        count_nonzero_per_row = tf.reduce_sum(tf.to_int32(nonzero_elements), axis=1)
        return tf.add(tf.reduce_sum(self.sequence_lengths, 1), tf.scalar_mul(2, count_nonzero_per_row))

    def forward(self, hidden_dropout_keep_prob, input_dropout_keep_prob, middle_dropout_keep_prob, reuse=True):
        word_embeddings = tf.nn.embedding_lookup(self.w_e, self.input_x1)

        with tf.variable_scope("forward", reuse=reuse):
            input_list = [word_embeddings]
            input_size = self.embedding_size
            if self.use_characters:
                input_list.append(self.char_embeddings)
                input_size += self.char_size
            if self.use_shape:
                shape_embeddings_shape = (self.shape_domain_size - 1, self.shape_size)
                w_s = tf_utils.initialize_embeddings(shape_embeddings_shape, name="w_s")
                shape_embeddings = tf.nn.embedding_lookup(w_s, self.input_x2)
                input_list.append(shape_embeddings)
                input_size += self.shape_size

            input_feats = tf.concat(axis=2, values=input_list)
            # self.input_feats_expanded = tf.expand_dims(self.input_feats, 1)
            input_feats_expanded_drop = tf.nn.dropout(input_feats, input_dropout_keep_prob)

            total_output_width = 2*self.hidden_dim

            with tf.name_scope("bilstm"):
                # selected_col_embeddings = tf.nn.embedding_lookup(token_embeddings, self.token_batch)
                fwd_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True, reuse=reuse)
                bwd_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True, reuse=reuse)
                lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell, dtype=tf.float32,
                                                                 inputs=input_feats_expanded_drop,
                                                                 parallel_iterations=50,
                                                                 sequence_length=self.flat_sequence_lengths)
                hidden_outputs = tf.concat(axis=2, values=lstm_outputs)

            h_concat_flat = tf.reshape(hidden_outputs, [-1, total_output_width])

            # Add dropout
            with tf.name_scope("middle_dropout"):
                h_drop = tf.nn.dropout(h_concat_flat, middle_dropout_keep_prob)

            # second projection
            with tf.name_scope("tanh_proj"):
                w_tanh = tf_utils.initialize_weights([total_output_width, self.hidden_dim], "w_tanh", init_type="xavier")
                b_tanh = tf.get_variable(initializer=tf.constant(0.01, shape=[self.hidden_dim]), name="b_tanh")
                self.l2_loss += tf.nn.l2_loss(w_tanh)
                self.l2_loss += tf.nn.l2_loss(b_tanh)
                h2_concat_flat = tf.nn.xw_plus_b(h_drop, w_tanh, b_tanh, name="h2_tanh")
                h2_tanh = tf_utils.apply_nonlinearity(h2_concat_flat, self.nonlinearity)

            # Add dropout
            with tf.name_scope("hidden_dropout"):
                h2_drop = tf.nn.dropout(h2_tanh, hidden_dropout_keep_prob)

        return h2_drop

    @lazy_property
    def lstm_output(self):
        return self.forward(self.hidden_dropout_keep_prob, self.input_dropout_keep_prob, self.middle_dropout_keep_prob, reuse=False)

    @lazy_property
    def lstm_output_no_drop(self):
        return self.forward(1.0, 1.0, 1.0)

    def calc_unflat_scores(self, lstm_output, reuse=True):
        with tf.variable_scope("forward", reuse=reuse):
            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                w_o = tf_utils.initialize_weights([self.hidden_dim, self.num_classes], "w_o", init_type="xavier")
                b_o = tf.get_variable(initializer=tf.constant(0.01, shape=[self.num_classes]), name="b_o")
                self.l2_loss += tf.nn.l2_loss(w_o)
                self.l2_loss += tf.nn.l2_loss(b_o)
                scores = tf.nn.xw_plus_b(lstm_output, w_o, b_o, name="scores")
                unflat_scores = tf.reshape(scores, tf.stack([self.batch_size, self.max_seq_len, self.num_classes]))

        return unflat_scores

    @lazy_property
    def unflat_scores(self):
        return self.calc_unflat_scores(self.lstm_output, reuse=False)

    @lazy_property
    def unflat_no_dropout_scores(self):
        return self.calc_unflat_scores(self.lstm_output_no_drop)

    @lazy_property
    def loss(self):
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            labels = tf.cast(self.input_y, 'int32')
            if self.viterbi:
                log_likelihood, transition_params = crf.crf_log_likelihood(self.unflat_scores, labels, self.label_mask, self.flat_sequence_lengths, transition_params=self.transition_params)
                # self.transition_params = transition_params
                loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unflat_scores, labels=labels)
                masked_losses = tf.multiply(losses, self.input_mask)
                loss = tf.div(tf.reduce_sum(masked_losses), tf.reduce_sum(self.input_mask))
            loss += self.l2_penalty * self.l2_loss

            drop_loss = tf.nn.l2_loss(tf.subtract(self.unflat_scores, self.unflat_no_dropout_scores))
            loss += self.drop_penalty * drop_loss

        return loss

    @lazy_property
    def test(self):
        # Calculate mean cross-entropy loss
        with tf.name_scope("test"):
            labels = tf.cast(self.input_y, 'int32')
            inputs, tag_masks, combined = crf.test(self.unflat_scores, labels, self.label_mask, self.flat_sequence_lengths, transition_params=self.transition_params)

            return inputs, tag_masks, combined

    @lazy_property
    def predictions(self):
        # Accuracy
        with tf.name_scope("predictions"):
            if self.viterbi:
                predictions = self.unflat_scores
            else:
                predictions = tf.argmax(self.unflat_scores, 2)

        return predictions
