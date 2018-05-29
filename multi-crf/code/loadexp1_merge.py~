from process2 import DataProcessor
from bilstm_3_sets import BiLSTM
from bilstm_char import BiLSTMChar
import tf_utils
import eval_f1 as evaluation
import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--hiddensize', default=200, type=int)
parser.add_argument('-p', '--dropout', default=.25, type=float)
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

data_path = '/mnt/nfs/work1/696ds-s18/ngreenberg/disjoint-dataset/data/'

cdr_path = data_path + 'cdr/'
bc_path = data_path + 'bc/'

embeddings_file = data_path + 'embeddings/glove.6B/glove.6B.100d.txt'

dp = DataProcessor(vocab=embeddings_file, window_size=window_size)

dp.read_file(cdr_path + 'ner_CID_Training_mine_PubTator.txt', 'cdr_train_weak', 'weak', update=True)

dp.read_file(cdr_path + 'ner_CDR_TrainingSet.PubTator.txt', 'cdr_train_gold', 'cdr', update=True)
dp.read_file(cdr_path + 'ner_CDR_DevelopmentSet.PubTator.txt', 'cdr_dev', 'cdr')
dp.read_file(cdr_path + 'ner_CDR_TestSet.PubTator.txt', 'cdr_test', 'cdr')

dp.read_file(bc_path + 'ner_CDR_train.txt', 'bc_train', 'bc', update=True)
dp.read_file(bc_path + 'ner_CDR_dev.txt', 'bc_dev', 'bc')
dp.read_file(bc_path + 'ner_CDR_test.txt', 'bc_test', 'bc')

###############
# build model #
###############

vocab_size = len(dp.token_map)
labels_weak_size = len(dp.label_maps['weak'])
labels_cdr_size = len(dp.label_maps['cdr'])
labels_bc_size = len(dp.label_maps['bc'])
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
    num_classes_A=labels_weak_size,
    num_classes_B=labels_cdr_size,
    num_classes_C=labels_bc_size,
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

outside_set = ["O", "<PAD>",  "<S>",  "</S>", "<ZERO>"]

for label, id in dp.label_maps['weak'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_A:
        type_set_A[label_type] = len(type_set_A)
    type_int_int_map_A[id] = type_set_A[label_type]

for label, id in dp.label_maps['cdr'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_B:
        type_set_B[label_type] = len(type_set_B)
    type_int_int_map_B[id] = type_set_B[label_type]

for label, id in dp.label_maps['bc'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_C:
        type_set_C[label_type] = len(type_set_C)
    type_int_int_map_C[id] = type_set_C[label_type]

###############
# train model #
###############

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

    inv_label_map_A = dp.inv_label_maps()['weak']
    inv_label_map_B = dp.inv_label_maps()['cdr']
    inv_label_map_C = dp.inv_label_maps()['bc']

    predictions = (predictions_A, predictions_B, predictions_C)
    type_sets = (type_set_A, type_set_B, type_set_C)
    type_int_int_maps = (type_int_int_map_A, type_int_int_map_B, type_int_int_map_C)
    inv_label_maps = (inv_label_map_A, inv_label_map_B, inv_label_map_C)

    merged_preds_list = []
    for (preds_A, preds_B, preds_C) in zip(predictions_A, predictions_B, predictions_C):
        merged_preds_sublist = []
        for (pred_A, pred_B, pred_C) in zip(preds_A, preds_B, preds_C):
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
                    possible = []

                    set_A = set(A[mention_start:i])
                    set_A.discard('O')
                    set_B = set(B[mention_start:i])
                    set_B.discard('O')
                    set_C = set(C[mention_start:i])
                    set_C.discard('O')

                    if len(set_A) != 0:
                        possible.append(A[mention_start:i])
                    if len(set_B) != 0:
                        possible.append(B[mention_start:i])
                    if len(set_C) != 0:
                        possible.append(C[mention_start:i])

                    ret[mention_start:i] = possible[np.random.randint(3) % len(possible)]

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

            merged_preds = [dp.label_maps['weak'][p] for p in ret]
            merged_preds_sublist.append(merged_preds)
        merged_preds_list.append(merged_preds_sublist)

    for i in range(len(batches_with_mask)):
        for j in range(len(batches_with_mask[i][1])):
            for k in range(len(batches_with_mask[i][1][j])):
                if batch_dataset == 'B':
                    inv_label_map = inv_label_map_B
                if batch_dataset == 'C':
                    inv_label_map = inv_label_map_C

                l = batches_with_mask[i][1][j][k]
                batches_with_mask[i][1][j][k] = dp.label_maps['weak'][inv_label_map[l]]

    f1_micro, precision = evaluation.segment_eval(
        batches_with_mask, merged_preds_list, type_set_A, type_int_int_map_A,
        inv_label_map_A, dp.inv_token_map(),
        outside_idx=map(
            lambda t: type_set_A[t] if t in type_set_A else type_set_A['O'], outside_set),
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


test_batches_B = dp.get_batches('cdr_test', batch_size, random=False)
test_batches_C = dp.get_batches('bc_test', batch_size, random=False)

model_path = '/mnt/nfs/work1/mccallum/tbansal/nathan/disjoint-dataset/models/'
save_path = model_path + 'exp1_%d_%.1f.ckpt' % (hidden_size, args.dropout)

saver.restore(sess, save_path)
print('\nModel loaded from %s' % save_path)

print('Best model results:')

run_evaluation(test_batches_B, 'B', 'cdr_test')
run_evaluation(test_batches_C, 'C', 'bc_test')
