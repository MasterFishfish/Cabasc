import tensorflow as tf

def last_relevant(inputs, lengths):
    # inputs [batch_size, steps, num_units]
    # length [batch_size] 储存各句子的长度
    lengths = tf.cast(lengths, tf.int32)
    batch_size = tf.shape(inputs)[0]
    steps = tf.shape(inputs)[1]
    edims = tf.shape(inputs)[2]
    index = tf.range(0, batch_size) * steps + (lengths - 1)
    flat = tf.reshape(inputs, [-1, edims])
    gather = tf.gather(flat, index)

def last_relevant1(inputs, lengths):
    pass

def relevant(inputs, lengths):
    pass