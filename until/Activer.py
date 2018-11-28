import tensorflow as tf

def activer(vec, active):
    # 内置tanh, relu两种激励函数
    # 当active输入的函数不包括在这两者之间的时候
    # 默认使用的tanh函数
    switch = {
        "tanh": tanh,
        "relu": relu,
    }
    func = switch.get(active, tanh)
    return func(vec)

def relu(vec):
    return tf.nn.relu(vec)

def tanh(vec):
    return tf.nn.tanh(vec)