import tensorflow as tf
from model import BiLSTM

vocab_size = 10
num_classes_A = 5
num_classes_B = 2

data = tf.placeholder(tf.int32, [None, None])
target_A = tf.placeholder(tf.float32, [None, None, num_classes_A])
target_B = tf.placeholder(tf.float32, [None, None, num_classes_B])
current_target = tf.placeholder(tf.string)

model = BiLSTM(data, target_A, target_B, vocab_size)

x = [[0, 4, 1, 5, 9],
     [3, 4, 6, 2, 7]]

y_A = [[[1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1]],

       [[0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]]]

y_B = [[[0, 1],
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0]],

       [[0, 1],
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0]]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    print('-' * 20)
    print(sess.run(model.cost_A, feed_dict={data: x, target_A: y_A}))
    print(sess.run(model.cost_B, feed_dict={data: x, target_B: y_B}))
    sess.run(model.optimize_A, feed_dict={data: x, target_A: y_A})

    print('-' * 20)
    print(sess.run(model.cost_A, feed_dict={data: x, target_A: y_A}))
    print(sess.run(model.cost_B, feed_dict={data: x, target_B: y_B}))
    sess.run(model.optimize_B, feed_dict={data: x, target_B: y_B})
