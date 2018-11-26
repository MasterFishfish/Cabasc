import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # g1 = tf.Graph()
    # with g1.as_default():
    #     a = tf.Variable(initial_value=1)
    #     a = tf.Variable(initial_value=2)
    #     a = tf.get_variable(shape=[2, 3], name="a_random")
    # # 在这个图中，究竟有多少个节点，a 是直接添加新的节点还是取代原来的节点
    # print(g1.as_graph_def())
    # with tf.Session(graph=g1) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(a))
    # b = tf.get_variable(shape=(2, 2, 3), dtype=tf.int32, name="b")
    # b = tf.cast(b, tf.float32)
    # cell = tf.nn.rnn_cell.GRUCell(num_units=12)
    # cells = tf.nn.rnn_cell.MultiRNNCell(cells=[cell])
    # output, _ = tf.nn.dynamic_rnn(cell=cells,dtype=tf.float32,sequence_length=[2, 1],inputs=b)
    # print(output)
    lengths = tf.Variable(initial_value=[6, 4, 5])
    batch_size=tf.Variable(initial_value=3)
    time_step = tf.Variable(initial_value=6)
    a = tf.range(0, batch_size)
    index = a * time_step + (lengths - 1)

    b = [[1, 2, 3],[2, 3, 4]]
    tf.cast(b, tf.float32)
    bv = tf.Variable(initial_value=b, name="b")
    # batch_size = 2, edims = 3
    bv_shaped = tf.reshape(bv, shape=[-1, 3, 1])
    # bv = tf.Variable(initial_value=b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))
        print(sess.run(bv_shaped))