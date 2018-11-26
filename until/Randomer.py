# 该类定义一个取得随机序列的类
import tensorflow as tf

class Randomer():

    stddev = 0

    # 类方法，访问类变量
    # 该类方法返回一个wshape形状的序列
    @staticmethod
    def random_normal(wshape):
        return tf.random_normal(shape=wshape, stddev=Randomer.stddev)

    # 类方法，更新类变量
    # 该类方法传入一个std来更新类里事先设定好的stddev
    @staticmethod
    def set_stddev(std):
        Randomer.stddev = std
