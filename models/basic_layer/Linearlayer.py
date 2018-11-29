import tensorflow as tf
from until.Randomer import Randomer
class Linearlayer():
    # 非三维的前向传播的输入层
    def __init__(self, w_shap, stddev, active="tanh", params=None):
        self.w_shape = w_shap
        if params is None:
            self.weights = tf.Variable(
                initial_value=Randomer.random_normal(wshape=self.w_shape),
                trainable=True
            )
        else:
            self.weights = tf.Variable(
                initial_value=params["wline"],
                trainable=True
            )
        self.stddev = stddev
        self.activer = active

    def forward(self, inputs):
        vec = inputs
        # 如果是三维要进行下面的步骤
        # batch_size = tf.shape(inputs)[0]
        # w_0 = self.w_shape[0]
        # w_1 = self.w_shape[1]
        # weights = tf.reshape(
        #     tensor=tf.tile(self.weights, [batch_size, 1]),
        #     shape=[-1, w_0, w_1]
        # )
        res = tf.matmul(vec, self.weights)
        return res
