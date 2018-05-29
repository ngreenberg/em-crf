from process import DataProcessor
from bilstm import BiLSTM
import tensorflow as tf
import tf_utils
from data_utils import SeqBatcher

###########################################
# hyperparameters and training parameters #
###########################################

embedding_size = 100
char_size = 0
shape_size = 5
hidden_size = 150
nonlinearity = 'tanh'
viterbi = True

learning_rate=0.0005
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-4

hidden_dropout = 0.75
input_dropout = 0.5
middle_droupout = 1.0
word_droupout = 0.75

clip_norm = 5

batch_size = 32

################
# process data #
################

embeddings_file = '/home/nathan/Programming/research/data/embeddings/glove.6B/glove.6B.100d.txt'

dp = DataProcessor(vocab=embeddings_file)

dp.read_file('/home/nathan/Programming/research/data/cdr/ner_CDR_train.txt',
             '/home/nathan/Programming/research/sandbox/protos/cdr_train.proto',
             'cdr', update=True)

# dp.read_file('/home/nathan/Programming/research/data/cdr/ner_CDR_test.txt',
#              '/home/nathan/Programming/research/sandbox/protos/cdr_test.proto',
#              'cdr')

# dp.read_file('/home/nathan/Programming/research/data/cdr/ner_CDR_dev.txt',
#              '/home/nathan/Programming/research/sandbox/protos/cdr_dev.proto',
#              'cdr')

# dp.read_file('/home/nathan/Programming/research/data/BC_VI_Task5/ner_CDR_BC_VI_train.txt',
#              '/home/nathan/Programming/research/sandbox/protos/bc_train.proto',
#              'BC_VI_Task5', update=True)

# dp.read_file('/home/nathan/Programming/research/data/BC_VI_Task5/ner_CDR_BC_VI_test.txt',
#              '/home/nathan/Programming/research/sandbox/protos/bc_test.proto',
#              'bc')

# dp.read_file('/home/nathan/Programming/research/data/BC_VI_Task5/ner_CDR_BC_VI_dev.txt',
#              '/home/nathan/Programming/research/sandbox/protos/bc_dev.proto',
#              'BC_VI_Task5')

###############
# build model #
###############

vocab_size = len(dp.token_map)
labels_cdr_size = len(dp.label_maps['cdr'])
# labels_bc_size = len(dp.label_maps['bc'])
shape_domain_size = len(dp.shape_map)
char_domain_size = len(dp.char_map)

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
print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (embeddings_used, vocab_size, embeddings_used / vocab_size * 100))

char_embeddings = None

with tf.Graph().as_default():
    model = BiLSTM(
        num_classes=labels_cdr_size,
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

    ###############
    # train model #
    ###############

    global_step = tf.Variable(0, name='global_step')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1, beta2=beta2, epsilon=epsilon,
                                       name='optimizer')

    model_vars = tf.global_variables()

    grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, model_vars), clip_norm)
    train_op = optimizer.apply_gradients(zip(grads, model_vars), global_step=global_step)

    opt_vars = [optimizer.get_slot(s, n) for n in optimizer.get_slot_names()
                for s in model_vars
                if optimizer.get_slot(s, n) is not None]

    model_vars += opt_vars

    train_eval_batcher = SeqBatcher('/home/nathan/Programming/research/sandbox/protos/cdr_train.proto', batch_size,
                                    num_buckets=0, num_epochs=1)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_batch = sess.run(train_eval_batcher.next_batch_op)
    train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch = train_batch

    print(train_label_batch.shape)
    print(train_token_batch.shape)
    print(train_shape_batch.shape)
    print(train_char_batch.shape)
    print(train_seq_len_batch.shape)
    print(train_tok_len_batch.shape)
