from process2 import DataProcessor
from bilstm import BiLSTM
from bilstm_char import BiLSTMChar
import tf_utils
import eval_f1 as evaluation
import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--hiddensize', default=150, type=int)
parser.add_argument('-m', '--model', default='A/model_best_parameters', type=str)
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

window_size = 3
pad_width = 1

batch_size = 32

################
# process data #
################

data_path = 'datapath/'

cdr_path = '/iesl/data/meta/pubtator/ner_paper/processed/train_peng_10000/'
bc_path = data_path + 'BC_VI_Task5/ner_CDR_BC_VI_'

embeddings_file = data_path + 'embeddings/glove.6B/glove.6B.100d.txt'

dp = DataProcessor(vocab=embeddings_file, window_size=window_size)

dp.read_file(cdr_path + 'ner_CDR_TrainingSet.PubTator.txt', 'cdr_train', 'cdr', update=True)
dp.read_file(cdr_path + 'ner_CDR_DevelopmentSet.PubTator.txt', 'cdr_dev', 'cdr')
dp.read_file(cdr_path + 'ner_CDR_TestSet.PubTator.txt', 'cdr_test', 'cdr')

dp.read_file(bc_path + 'train.txt', 'bc_train', 'bc', update=True)
dp.read_file(bc_path + 'dev.txt', 'bc_dev', 'bc')
dp.read_file(bc_path + 'test.txt', 'bc_test', 'bc')

##############
# build model #
##############

vocab_size = len(dp.token_map)
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
print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (embeddings_used, vocab_size, embeddings_used / vocab_size * 100) + '\n')

char_embeddings = None
if char_size > 0:
    char_embedding_model = BiLSTMChar(
        char_domain_size, char_size, int(char_tok_size / 2))
    char_embeddings = char_embedding_model.outputs

model = BiLSTM(
    num_classes_A=labels_cdr_size,
    num_classes_B=labels_bc_size,
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

outside_set = ["O", "<PAD>",  "<S>",  "</S>", "<ZERO>"]

for label, id in dp.label_maps['cdr'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_A:
        type_set_A[label_type] = len(type_set_A)
    type_int_int_map_A[id] = type_set_A[label_type]

for label, id in dp.label_maps['bc'].items():
    label_type = label if label in outside_set else label[2:]
    if label_type not in type_set_B:
        type_set_B[label_type] = len(type_set_B)
    type_int_int_map_B[id] = type_set_B[label_type]

###############
# load model #
###############

saver = tf.train.Saver()

sess = tf.Session()

model_path = 'modelpath/%s.ckpt' % args.model
saver.restore(sess, model_path)
print('\nModel loaded from %s' % model_path)

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
    actual_seq_lens = np.add(np.sum(seq_len_batch, axis=1), pad_width * (num_sentences_batch + 1)).astype('int')
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

def run_evaluation(batches, extra_text='', A=True):
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
            if A:
                preds, transition_params = sess.run([model.predictions_A, model.transition_params_A], feed_dict=lstm_feed)
            else:
                preds, transition_params = sess.run([model.predictions_B, model.transition_params_B], feed_dict=lstm_feed)

            viterbi_repad = np.empty((batch_size, batch_seq_len))
            for i, (unary_scores, sequence_lens) in enumerate(zip(preds, seq_len_batch)):
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)
                viterbi_repad[i] = viterbi_sequence
            predictions.append(viterbi_repad)
        else:
            if A:
                preds, scores = sess.run([model.predictions_A, model.unflat_scores_A], feed_dict=lstm_feed)
            else:
                preds, scores = sess.run([model.predictions_B, model.unflat_scores_B], feed_dict=lstm_feed)

    if A:
        type_set = type_set_A
        type_int_int_map = type_int_int_map_A
    else:
        type_set = type_set_B
        type_int_int_map = type_int_int_map_B

    f1_micro, precision = evaluation.segment_eval(
        batches_with_mask, predictions, type_set, type_int_int_map,
        dp.inv_label_maps()['cdr'], dp.inv_token_map(),
        outside_idx=map(lambda t: type_set[t] if t in type_set else type_set['O'], outside_set),
        pad_width=pad_width, start_end=False,
        extra_text='Segment evaluation %s:' % extra_text)

    print('')

    return f1_micro, precision

train_batches_A = dp.get_batches('cdr_train', batch_size, random=False)
train_batches_B = dp.get_batches('bc_train', batch_size, random=False)

dev_batches_A = dp.get_batches('cdr_dev', batch_size, random=False)
dev_batches_B = dp.get_batches('bc_dev', batch_size, random=False)

test_batches_A = dp.get_batches('cdr_test', batch_size, random=False)
test_batches_B = dp.get_batches('bc_test', batch_size, random=False)

run_evaluation(train_batches_A, 'cdr_train', A=True)
run_evaluation(train_batches_B, 'bc_train', A=False)

run_evaluation(dev_batches_A, 'cdr_dev', A=True)
run_evaluation(dev_batches_B, 'bc_dev', A=False)

run_evaluation(test_batches_A, 'cdr_test', A=True)
run_evaluation(test_batches_B, 'bc_test', A=False)
