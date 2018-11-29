import tensorflow as tf
from until.SoftMask import softmax_mask

# 基于dot的相似度计算得到attention模型
class DotAttentionLayer():
    def __init__(self, edims):
        self.edims = edims

    def count_alpha(self, contexts, aspects, ctx_bitmap):
        mem_size = tf.shape(contexts)[1]
        ctx = contexts
        # asp.shape = [batch_size, edims]
        asp = aspects
        asp = tf.reshape(asp, [-1, self.edims, 1])
        # query * key shape = [batch_size, steps, 1]
        qk = tf.reshape(tf.matmul(ctx, asp), [-1, mem_size])
        # [batch_size, steps]
        alpha = softmax_mask(inputs=qk, seq_mask=ctx_bitmap)
        return alpha

    def forward(self, context, aspect, ctx_bitmap):
        mem_size = tf.shape(context)[1]
        context = context
        aspect = aspect
        # adjust attention
        alpha = self.count_alpha(context, aspect, ctx_bitmap)
        vec = tf.matmul(
            tf.reshape(alpha, [-1, 1, mem_size]),
            context
        )
        return vec, alpha