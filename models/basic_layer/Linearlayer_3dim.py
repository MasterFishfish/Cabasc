# Linearlayer_3dim
# 该类定义一个传统神经网路的输入层
import tensorflow as tf
from until.Randomer import Randomer

class Linearlayer_3dim():

    def __init__(self,
                 w_shape,
                 stddev,
                 active='tanh',
                 param=None):
        # w_shape(list/tuple)该层的weights的形状
        # 输入的input.shape=[batch_size, steps, input_edims]
        # w_shape.shape=[input_edims, output_edims]
        # 在前向传播函数里边将w_shape其扩展到[batch_size, input_edims, output_edims]
        # stddev(float)该层的weights自动随机初始化的时候，random函数里面需要的参数
        # active(string)该层的激励函数？
        # param(dict)是否指定weights的值, 为一个字典{"wline":init_weights} 储存weights值
        # param 指定的weights形状必须为w_shape
        self.w_shape = w_shape
        self.stddev = stddev
        if param is None:
            self.weights = tf.Variable(
                initial_value=Randomer.random_normal(wshape=self.w_shape, stddev=self.stddev),
                trainable=True
            )
        else:
            self.weights = tf.Variable(
                initial_value=param["wline"],
                trainable=True
            )
        self.activer = active

    def forward(self, inputs):
        # inputs(tensor): 为该层的输入，shape=[batch_size, steps, edims]
        # 这一个函数将要计算 input * weight 之后的值
        batch_size = tf.shape(inputs)[0]
        steps = tf.shape(inputs)[1]

        # input_edims = tf.shape(self.weights)[0]
        input_edims = tf.shape(inputs)[2]
        output_edims = tf.shape(self.weights)[1]
        batch_weights = tf.reshape(
            tensor=tf.tile(self.weights, [batch_size, 1]),
            shape=[batch_size, input_edims, output_edims])
        output = tf.matmul(inputs, batch_weights)
        return output


