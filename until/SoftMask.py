import tensorflow as tf

def normalizer(norm_type,
               inputs,
               seq_mask, axis=1):
    # inputs.shape = [batch_size, steps]
    switch = {
        "softmax": softmax_mask,
        "alpha": alpha_mask,
        "sigmoid": sigmoid_mask,
        "sigmoid2": sigmoid2_mask,
        "none": none_mask
    }
    func = switch.get(norm_type, none_mask)
    return func(inputs, seq_mask, axis=1)

def none_mask(inputs, seq_mask, axis=1):
    return inputs * seq_mask

def softmax_mask(inputs, seq_mask, axis=1):
    # mask用于移除padding的影响，(padding从何而来？？)
    # seq_mask.shape == inputs.shape

    inputs = tf.cast(inputs, tf.float32)
    seq_mask = tf.cast(seq_mask, tf.float32)
    inputs = inputs * seq_mask
    max_nums = tf.reduce_max(inputs, axis=axis, keep_dims=True)

    # inputs - max_nums 是一种归一化方法？
    inputs = tf.exp(inputs - max_nums)
    # inputs * mask 究竟避免了什么
    inputs = inputs * seq_mask
    _sum = tf.reduce_sum(inputs, axis=axis, keep_dims=True) + 1e-9
    return inputs / _sum

def alpha_mask(inputs, seq_mask, axis=1):
    pass

def sigmoid_mask(inputs, seq_mask, axis=1):
    pass

def sigmoid2_mask(inputs, seq_mask, axis=1):
    pass