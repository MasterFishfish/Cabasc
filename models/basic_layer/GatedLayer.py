import tensorflow as tf
from until.Randomer import Randomer

class GatedLayer():
    def __init__(self, hidden_size, stddev=None):
        self.hidden_size = hidden_size
        self.stddev = stddev
        self.W = tf.Variable(
            Randomer.random_normal(
                [self.hidden_size, self.hidden_size]
            ), trainable=True
        )
        self.U = tf.Variable(
            Randomer.random_normal(
                [self.hidden_size, self.hidden_size]
            ), trainable=True
        )
        self.bias = tf.Variable(tf.zeros([1]), trainable=True)

    def count_layer(self, xa, xb):
        # 计算门限
        batch_size = tf.shape(xa)[0]
        W_3dim = tf.reshape(
            tf.tile(self.W, [batch_size, 1]),
            [batch_size, self.hidden_size, self.hidden_size]
        )
        U_3dim = tf.reshape(
            tf.tile(self.U, [batch_size, 1]),
            [batch_size, self.hidden_size, self.hidden_size]
        )
        alpha = tf.add(
            tf.add(
                tf.matmul(xa, W_3dim),
                tf.matmul(xb, U_3dim)
            ),
            self.b
        )
        alpha = tf.sigmoid(alpha)

        return alpha

    def forward(self, xa, xb, xc=None):
        alpha = self.count_layer(xa, xb)
        return ( 1 - alpha ) * xa + alpha * xb