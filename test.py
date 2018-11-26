import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    a = [[[1, 2],
          [2, 3],
          [4, 5]],
         [[4, 7],
          [5, 0],
          [4, 8]],
         [[2, 9],
          [3, 5],
          [4, 1]]]
    # batch_size = 3, steps = 3, edims = 2
    # batch_size = 3, edims = 2, edims = 2
    b = [[[2, 1],
          [2, 1]],
         [[2, 1],
          [2, 1]],
         [[2, 1],
          [2, 1]]]
    d = [[1],
         [2]]
    c = tf.to_float(tf.matmul(a, b))
    c1 = tf.nn.tanh(c)
    # c [batch_size, steps, edims]
    # weight_c [batch_size, edims, 1]
    # c * weight_c [batch_size, steps, 1]
    weight_c = [[[2],
                 [2]],
                [[3],
                 [3]],
                [[1],
                 [1]]]
    weight_c = tf.to_float(weight_c)
    result_c = tf.matmul(c1, weight_c)
    c_re = tf.reshape(result_c, [-1, 3])
    c_print = tf.Variable(initial_value=c_re, name="print_c")
    seq_mask = [[0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]]
    seq_mask = tf.cast(seq_mask, dtype=tf.float32)
    c_re = c_re * seq_mask
    max_num = tf.reduce_max(c_re, axis=1, keepdims=True)
    a = c_re - max_num
    c_re = tf.exp(c_re - max_num)
    outputs_re = c_re * seq_mask
    thesum = tf.reduce_sum(outputs_re, axis=1, keepdims=True) + 1e-9
    output = outputs_re / thesum

    #steps = 3 edim = 2
    #output_alpha0 = tf.reshape(output, shape=[-1, 1, 3])
    output_alpha = tf.reshape(output, shape=[-1, 3, 1])
    output_alpha_3dim = tf.tile(output_alpha, [1, 1, 2])
    # f = [[[1], [2], [3]],
    #      [[4], [5], [6]],
    #      [[3], [8], [0]]]
    # g = [[[2], [1]],
    #      [[2], [1]],
    #      [[2], [1]]]
    #h = tf.matmul(f, g)
    e = tf.reshape(tf.tile(d, [3, 1]), [3, 2, 1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c_print))

        print("==============================================")
        print("alpha:")
        print(sess.run(output))
        print("==============================================")
        print("output_alpha:")
        print(sess.run(output_alpha))
        print("==============================================")
        print("output_alpha_3dim:")
        print(sess.run(output_alpha_3dim))