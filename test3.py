import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    a = [[1, 2, 3, 4],
         [2, 3, 4, 5]]
    av = tf.Variable(initial_value=a, name="av")
    b = [1, 1, 1, 1]
    bv = tf.Variable(initial_value=b, name="bv")
    cv = tf.add(av, bv)
    ev = bv
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(cv))
        print(sess.run(tf.tile(av, [2, 1])))
        print(sess.run(av))
        print(sess.run(ev))
