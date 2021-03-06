from process2 import DataProcessor
from bilstm_3_sets import BiLSTM
from bilstm_char import BiLSTMChar
import tf_utils
import eval_f1 as evaluation
import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--hiddensize', default=150, type=int)
parser.add_argument('-p', '--dropout', default=.3, type=float)
args = parser.parse_args()

###########################################
# hyperparameters and training parameters #
###########################################

embedding_size = 100
char_size = 25
char_tok_size = 50
shape_size = 5
hidden_size = args.hiddensize
nonlinearity = 'tanh'
viterbi = True

learning_rate = 0.0005
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

hidden_dropout = args.dropout
input_dropout = args.dropout
middle_dropout = args.dropout

word_dropout_keep = 0.5
char_input_dropout_keep = .9

l2 = 0.0
regularize_drop_penalty = 1e-4

clip_norm = 10

window_size = 3
pad_width = 1

batch_size = 32

################
# process data #
################

data_path = 'datapath/'

path = data_path + 'pubmed/'

embeddings_file = data_path + 'embeddings/glove.6B/glove.6B.100d.txt'

dp = DataProcessor(vocab=embeddings_file, window_size=window_size)

dp.read_file(path + 'train_split_1_modified', 'A_train', 'A', update=True)
dp.read_file(path + 'train_split_2_modified', 'B_train', 'B', update=True)
dp.read_file(path + 'train_split_3_modified', 'C_train', 'C', update=True)

dp.read_file(path + 'ner_dev', 'dev', 'full')
dp.read_file(path + 'ner_test', 'test', 'full')

###############
# build model #
###############

vocab_size = len(dp.token_map)
labels_A_size = len(dp.label_maps['A'])
labels_B_size = len(dp.label_maps['B'])
labels_C_size = len(dp.label_maps['C'])
labels_full_size = len(dp.label_maps['full'])
shape_domain_size = len(dp.shape_map)
char_domain_size = len(dp.char_map)

print('Loading embeddings from ' + embeddings_file)
embeddings_shape = (vocab_size - 1, embedding_size)
embeddings = tf_utils.embedding_values(embeddings_shape, old=False)
embeddings_used = 0
with open(embeddings_file, 'r') as f:
    for line in f.readlines():
        split_line = line.strip().split(' ')
        word = split_line[0]
        embedding = split_line[1:]
        embeddings[dp.token_map[word] - 1] = list(map(float, embedding))
        embeddings_used += 1
print("Loaded %d/%d embeddings (%2.2f%% coverage)" %
      (embeddings_used, vocab_size, embeddings_used / vocab_size * 100) + '\n')

if char_size > 0:
    char_embedding_model = BiLSTMChar(
        char_domain_size, char_size, int(char_tok_size / 2))
    char_embeddings = char_embedding_model.outputs

model = BiLSTM(
    num_classes_A=labels_A_size,
    num_classes_B=labels_B_size,
    num_classes_C=labels_C_size,
    vocab_size=vocab_size,
    shape_domain_size=shape_domain_size,
    char_domain_size=char_domain_size,
    char_size=char_size,
    embedding_size=embedding_size,
    shape_size=shape_size,
    nonlinearity=nonlinearity,
    viterbi=viterbi,
    hidden_dim=hidden_size,
    char_embeddings=char_embeddings,
    embeddings=embeddings)

type_set_A = {}
type_int_int_map_A = {}

type_set_B = {}
type_int_int_map_B = {}

type_set_C = {}
type_int_int_map_C = {}

type_set_full = {}
type_int_int_map_full = {}

outside_set = ["O", "<PAD>",  "<S>",  "</S>", "<ZERO>"]

for label, id in dp.label_maps['A'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_A:
        type_set_A[label_type] = len(type_set_A)
    type_int_int_map_A[id] = type_set_A[label_type]

for label, id in dp.label_maps['B'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_B:
        type_set_B[label_type] = len(type_set_B)
    type_int_int_map_B[id] = type_set_B[label_type]

for label, id in dp.label_maps['C'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_C:
        type_set_C[label_type] = len(type_set_C)
    type_int_int_map_C[id] = type_set_C[label_type]

for label, id in dp.label_maps['full'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_full:
        type_set_full[label_type] = len(type_set_full)
    type_int_int_map_full[id] = type_set_full[label_type]

###############
# train model #
###############

global_step = tf.Variable(0, name='global_step')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta1, beta2=beta2, epsilon=epsilon,
                                   name='optimizer')

model_vars = tf.global_variables()

grads_A, _ = tf.clip_by_global_norm(
    tf.gradients(model.loss_A, model_vars), clip_norm)
train_op_A = optimizer.apply_gradients(
    zip(grads_A, model_vars), global_step=global_step)

grads_B, _ = tf.clip_by_global_norm(
    tf.gradients(model.loss_B, model_vars), clip_norm)
train_op_B = optimizer.apply_gradients(
    zip(grads_B, model_vars), global_step=global_step)

grads_C, _ = tf.clip_by_global_norm(
    tf.gradients(model.loss_C, model_vars), clip_norm)
train_op_C = optimizer.apply_gradients(
    zip(grads_C, model_vars), global_step=global_step)

saver = tf.train.Saver(max_to_keep=1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def mask(batch):
    token_batch, label_batch, shape_batch, _, seq_len_batch, _ = batch

    # make mask out of seq lens
    _, batch_seq_len = token_batch.shape

    max_sentences = max(map(len, seq_len_batch))
    new_seq_len_batch = np.zeros((batch_size, max_sentences))
    for i, seq_len_list in enumerate(seq_len_batch):
        new_seq_len_batch[i, :len(seq_len_list)] = seq_len_list
    seq_len_batch = new_seq_len_batch
    num_sentences_batch = np.sum(seq_len_batch != 0, axis=1)

    mask_batch = np.zeros((batch_size, batch_seq_len)).astype('int')
    actual_seq_lens = np.add(np.sum(seq_len_batch, axis=1),
                             pad_width * (num_sentences_batch + 1)).astype('int')
    for i, seq_len in enumerate(actual_seq_lens):
        mask_batch[i, :seq_len] = 1

    return batch_seq_len, mask_batch, seq_len_batch


def char_feed(token_batch, char_batch, tok_len_batch, dropout_keep_prob=1.0):
    batch_size, batch_seq_len = token_batch.shape

    char_lens = np.sum(tok_len_batch, axis=1)
    max_char_len = np.max(tok_len_batch)
    padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))

    for b in range(batch_size):
        char_indices = [item for sublist in
                        [range(i * max_char_len, i * max_char_len + d)
                         for i, d in enumerate(tok_len_batch[b])]
                        for item in sublist]
        padded_char_batch[b, char_indices] = char_batch[b][:char_lens[b]]

    if char_size == 0:
        char_embedding_feeds = {}
    else:
        char_embedding_feeds = {
            char_embedding_model.input_chars: padded_char_batch,
            char_embedding_model.batch_size: batch_size,
            char_embedding_model.max_seq_len: batch_seq_len,
            char_embedding_model.token_lengths: tok_len_batch,
            char_embedding_model.max_tok_len: max_char_len,
            char_embedding_model.input_dropout_keep_prob: dropout_keep_prob
        }

    return char_embedding_feeds


def run_evaluation(batches, batch_dataset, extra_text=''):
    probabilities_A = []
    probabilities_B = []
    probabilities_C = []

    predictions_A = []
    predictions_B = []
    predictions_C = []
    batches_with_mask = []
    for batch in batches:
        token_batch, label_batch, shape_batch, char_batch, seq_len_batch, tok_len_batch = batch
        batch_seq_len, mask_batch, seq_len_batch = mask(batch)
        batches_with_mask.append(batch + (mask_batch,))

        char_embedding_feed = char_feed(token_batch, char_batch, tok_len_batch)
        lstm_feed = {
            model.input_x1: token_batch,
            model.input_x2: shape_batch,
            model.input_y: label_batch,
            model.input_mask: mask_batch,
            model.sequence_lengths: seq_len_batch,
            model.max_seq_len: batch_seq_len,
            model.batch_size: batch_size,
        }
        lstm_feed.update(char_embedding_feed)

        if viterbi:
            [probs_A], [probs_B], [probs_C] = sess.run(model.marginal_probabilities(), feed_dict=lstm_feed)
            probabilities_A.append(probs_A)
            probabilities_B.append(probs_B)
            probabilities_C.append(probs_C)

            run_list_A = [model.predictions_A, model.transition_params_A]
            run_list_B = [model.predictions_B, model.transition_params_B]
            run_list_C = [model.predictions_C, model.transition_params_C]

            preds_A, transition_params_A = sess.run(run_list_A, feed_dict=lstm_feed)

            viterbi_repad_A = np.empty((batch_size, batch_seq_len))
            for i, (unary_scores, sequence_lens) in enumerate(zip(preds_A, seq_len_batch)):
                viterbi_sequence_A, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores, transition_params_A)
                viterbi_repad_A[i] = viterbi_sequence_A
            predictions_A.append(viterbi_repad_A)

            preds_B, transition_params_B = sess.run(run_list_B, feed_dict=lstm_feed)

            viterbi_repad_B = np.empty((batch_size, batch_seq_len))
            for i, (unary_scores, sequence_lens) in enumerate(zip(preds_B, seq_len_batch)):
                viterbi_sequence_B, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores, transition_params_B)
                viterbi_repad_B[i] = viterbi_sequence_B
            predictions_B.append(viterbi_repad_B)


            preds_C, transition_params_C = sess.run(run_list_C, feed_dict=lstm_feed)

            viterbi_repad_C = np.empty((batch_size, batch_seq_len))
            for i, (unary_scores, sequence_lens) in enumerate(zip(preds_C, seq_len_batch)):
                viterbi_sequence_C, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores, transition_params_C)
                viterbi_repad_C[i] = viterbi_sequence_C
            predictions_C.append(viterbi_repad_C)

        else:
            if batch_dataset == 'A':
                run_list = [model.predictions_A, model.unflat_scores_A]
            elif batch_dataset == 'B':
                run_list = [model.predictions_B, model.unflat_scores_B]
            elif batch_dataset == 'C':
                run_list = [model.predictions_C, model.unflat_scores_C]

            preds, scores = sess.run(run_list, feed_dict=lstm_feed)

    inv_label_map_A = dp.inv_label_maps()['A']
    inv_label_map_B = dp.inv_label_maps()['B']
    inv_label_map_C = dp.inv_label_maps()['C']
    inv_label_map_full = dp.inv_label_maps()['full']

    predictions = (predictions_A, predictions_B, predictions_C)
    type_sets = (type_set_A, type_set_B, type_set_C)
    type_int_int_maps = (type_int_int_map_A, type_int_int_map_B, type_int_int_map_C)
    inv_label_maps = (inv_label_map_A, inv_label_map_B, inv_label_map_C)

    merged_preds_list = []
    for (preds_A, preds_B, preds_C, probs_A, probs_B, probs_C) in zip(predictions_A, predictions_B, predictions_C, probabilities_A, probabilities_B, probabilities_C):
        merged_preds_sublist = []
        for (pred_A, pred_B, pred_C, prob_A, prob_B, prob_C) in zip(preds_A, preds_B, preds_C, probs_A, probs_B, probs_C):
            A = [inv_label_map_A[p] for p in pred_A]
            B = [inv_label_map_B[p] for p in pred_B]
            C = [inv_label_map_C[p] for p in pred_C]

            def beginning(p):
                return p[0] == 'U' or p[0] == 'B'

            ret = A.copy()

            in_conflict = False
            mention_start = 0
            for i, (a, b, c) in enumerate(zip(A, B, C)):
                if in_conflict and ((a == 'O' and b == 'O' and c == 'O') or (beginning(a) and beginning(b) and beginning(c))):
                    set_A = set(A[mention_start:i])
                    set_A.discard('O')
                    set_B = set(B[mention_start:i])
                    set_B.discard('O')
                    set_C = set(C[mention_start:i])
                    set_C.discard('O')

                    A_total = B_total = C_total = -np.inf
                    if len(set_A) != 0:
                        A_total = 0
                    if len(set_B) != 0:
                        B_total = 0
                    if len(set_C) != 0:
                        C_total = 0

                    for x in range(mention_start, i):
                        A_total += prob_A[x, int(pred_A[x])]
                        B_total += prob_B[x, int(pred_B[x])]
                        C_total += prob_C[x, int(pred_C[x])]

                    ret[mention_start:i] = [A[mention_start:i], B[mention_start:i], C[mention_start:i]][np.argmax([A_total, B_total, C_total])]

                    in_conflict = False

                if not in_conflict and (beginning(a) or beginning(b) or beginning(c)):
                    mention_start = i

                in_conflict = in_conflict or a != b or a != c or b != c

                # options = set([A[i], B[i], C[i]])
                # options.discard('O')
                # if options == set():
                #     merged_preds.append('O')
                # else:
                #     merged_preds.append(np.random.choice(list(options)))

            merged_preds = [dp.label_maps['full'][p] for p in ret]
            merged_preds_sublist.append(merged_preds)
        merged_preds_list.append(merged_preds_sublist)

    for i in range(len(batches_with_mask)):
        for j in range(len(batches_with_mask[i][1])):
            for k in range(len(batches_with_mask[i][1][j])):
                if batch_dataset == 'A':
                    inv_label_map = inv_label_map_A
                if batch_dataset == 'B':
                    inv_label_map = inv_label_map_B
                if batch_dataset == 'C':
                    inv_label_map = inv_label_map_C
                if batch_dataset == 'full':
                    inv_label_map = inv_label_map_full

                l = batches_with_mask[i][1][j][k]
                batches_with_mask[i][1][j][k] = dp.label_maps['full'][inv_label_map[l]]

    f1_micro, precision = evaluation.segment_eval(
        batches_with_mask, merged_preds_list, type_set_full, type_int_int_map_full,
        inv_label_map_full, dp.inv_token_map(),
        outside_idx=map(
            lambda t: type_set_full[t] if t in type_set_full else type_set_full['O'], outside_set),
        pad_width=pad_width, start_end=False,
        extra_text='Segment evaluation %s:' % extra_text)

    print('')

    return f1_micro, precision


def run_evaluation_old(batches, batch_dataset, extra_text=''):
    predictions = []
    batches_with_mask = []
    for batch in batches:
        token_batch, label_batch, shape_batch, char_batch, seq_len_batch, tok_len_batch = batch
        batch_seq_len, mask_batch, seq_len_batch = mask(batch)
        batches_with_mask.append(batch + (mask_batch,))

        char_embedding_feed = char_feed(token_batch, char_batch, tok_len_batch)
        lstm_feed = {
            model.input_x1: token_batch,
            model.input_x2: shape_batch,
            model.input_y: label_batch,
            model.input_mask: mask_batch,
            model.sequence_lengths: seq_len_batch,
            model.max_seq_len: batch_seq_len,
            model.batch_size: batch_size,
        }
        lstm_feed.update(char_embedding_feed)

        if viterbi:
            if batch_dataset == 'A':
                run_list = [model.predictions_A, model.transition_params_A]
            elif batch_dataset == 'B':
                run_list = [model.predictions_B, model.transition_params_B]
            elif batch_dataset == 'C':
                run_list = [model.predictions_C, model.transition_params_C]

            preds, transition_params = sess.run(run_list, feed_dict=lstm_feed)

            viterbi_repad = np.empty((batch_size, batch_seq_len))
            for i, (unary_scores, sequence_lens) in enumerate(zip(preds, seq_len_batch)):
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    unary_scores, transition_params)
                viterbi_repad[i] = viterbi_sequence
            predictions.append(viterbi_repad)

        else:
            if batch_dataset == 'A':
                run_list = [model.predictions_A, model.unflat_scores_A]
            elif batch_dataset == 'B':
                run_list = [model.predictions_B, model.unflat_scores_B]
            elif batch_dataset == 'C':
                run_list = [model.predictions_C, model.unflat_scores_C]

            preds, scores = sess.run(run_list, feed_dict=lstm_feed)

    if batch_dataset == 'A':
        type_set = type_set_A
        type_int_int_map = type_int_int_map_A
        inv_label_map = dp.inv_label_maps()['A']
    elif batch_dataset == 'B':
        type_set = type_set_B
        type_int_int_map = type_int_int_map_B
        inv_label_map = dp.inv_label_maps()['B']
    elif batch_dataset == 'C':
        type_set = type_set_C
        type_int_int_map = type_int_int_map_C
        inv_label_map = dp.inv_label_maps()['C']
    elif batch_dataset == 'full':
        type_set = type_set_full
        type_int_int_map = type_int_int_map_full
        inv_label_map = dp.inv_label_maps()['full']

    f1_micro, precision = evaluation.segment_eval(
        batches_with_mask, predictions, type_set, type_int_int_map,
        inv_label_map, dp.inv_token_map(),
        outside_idx=map(
            lambda t: type_set[t] if t in type_set else type_set['O'], outside_set),
        pad_width=pad_width, start_end=False,
        extra_text='Segment evaluation %s:' % extra_text)

    print('')

    return f1_micro, precision


def train(batch, batch_dataset):
    token_batch, label_batch, shape_batch, char_batch, seq_len_batch, tok_len_batch = batch
    batch_seq_len, mask_batch, seq_len_batch = mask(batch)

    # apply word dropout
    # create word dropout mask
    word_probs = np.random.random(token_batch.shape)
    drop_indices = np.where((word_probs > word_dropout_keep) & (
        token_batch != dp.token_map['<PAD>']))
    token_batch[drop_indices[0], drop_indices[1]] = dp.token_map['<OOV>']

    char_embedding_feed = char_feed(
        token_batch, char_batch, tok_len_batch, char_input_dropout_keep)
    lstm_feed = {
        model.input_x1: token_batch,
        model.input_x2: shape_batch,
        model.input_y: label_batch,
        model.input_mask: mask_batch,
        model.sequence_lengths: seq_len_batch,
        model.max_seq_len: batch_seq_len,
        model.batch_size: batch_size,
        model.hidden_dropout_keep_prob: 1 - hidden_dropout,
        model.middle_dropout_keep_prob: 1 - middle_dropout,
        model.input_dropout_keep_prob: 1 - input_dropout,
        model.l2_penalty: l2,
        model.drop_penalty: regularize_drop_penalty
    }
    lstm_feed.update(char_embedding_feed)

    if batch_dataset == 'A':
        run_list = [train_op_A, model.loss_A]
    elif batch_dataset == 'B':
        run_list = [train_op_B, model.loss_B]
    elif batch_dataset == 'C':
        run_list = [train_op_C, model.loss_C]

    _, loss = sess.run(run_list, feed_dict=lstm_feed)

    return loss


epochs = 6000

train_batches_A = dp.get_batches('A_train', batch_size, random=False)
train_batches_B = dp.get_batches('B_train', batch_size, random=False)
train_batches_C = dp.get_batches('C_train', batch_size, random=False)

dev_batches = dp.get_batches('dev', batch_size, random=False)

test_batches = dp.get_batches('test', batch_size, random=False)

max_f1 = 0
epoch_last_saved = 0
last_saved = 'N/A'
for i in range(epochs):
    batches_A = dp.get_batches('A_train', batch_size)
    batches_B = dp.get_batches('B_train', batch_size)
    batches_C = dp.get_batches('C_train', batch_size)

    l = min(len(batches_A), len(batches_B), len(batches_C))

    for j in range(l):
        loss_A = train(batches_A[j], 'A')
        loss_B = train(batches_B[j], 'B')
        loss_C = train(batches_C[j], 'C')

        if j % 100 == 0:
            f1, _ = run_evaluation(
            	dev_batches, 'full', 'dev (iteration %d)' % i)

            model_path = 'modelpath/3_sets/'

            if f1 > max_f1:
                save_path = saver.save(
                    sess, model_path + 'exp1_split_%d_%.1f.ckpt' % (hidden_size, args.dropout))
                print('Model saved in file: %s\n' % save_path)
                max_f1 = f1
                last_saved = 'epoch %d, batch %d' % (i, j)
                epoch_last_saved = i
            else:
                print('Lasted saved at %s\n' % last_saved)

    if (i - epoch_last_saved) > 100:
        break

print('Training complete. Loading best model')

saver.restore(sess, save_path)
print('\nModel loaded from %s' % save_path)

print('Best model results:')

run_evaluation_old(train_batches_A, 'A', 'A_train')
run_evaluation_old(train_batches_B, 'B', 'B_train')
run_evaluation_old(train_batches_B, 'C', 'C_train')

run_evaluation(dev_batches, 'full', 'dev')
run_evaluation(test_batches, 'full', 'test')
