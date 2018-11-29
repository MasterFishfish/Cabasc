import tensorflow as tf
from until.SoftMask import softmax_mask
from Linearlayer_3dim import Linearlayer_3dim
from until.SoftMask import normalizer

class GenAttentionLayer():
    def __init__(self, edim,stddev,norm_type = 'softmax'):
        self.edim = edim
        self.line_layer = Linearlayer_3dim(
            [self.edim, self.edim],
            stddev
        )
        self.norm_type = norm_type
    def count_alpha(self, context, aspect, ctx_bitmap):
        # V * aspect
        mem_size = tf.shape(context)[1]
        context = context
        aspect = aspect
        # adjust attention
        asp_3dim = tf.reshape(aspect, [-1, self.edim, 1])
        # gout = tf.matmul(context, asp_3dim)
        res_act = self.line_layer.forward(context)
        res_act = tf.reshape(
            tf.matmul(res_act, asp_3dim),
            [-1, mem_size]
        )
        # alpha = tf.nn.softmax(tf.reshape(gout, [-1, mem_size]))
        alpha = normalizer(self.norm_type, res_act, ctx_bitmap, 1)
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

    def forward_max_pool(self, context, aspect, ctx_bitmap):
        # 最大值池化之后的sentences embeddings
        mem_size = tf.shape(context)[1]
        ctx = context
        asp = aspect
        alpha = self.count_alpha(context=ctx, aspect=asp, ctx_bitmap=ctx_bitmap)
        vec = tf.reshape(
            tf.reduce_max(
                tf.multiply(
                    tf.tile(
                        tf.reshape(alpha, shape=[-1, mem_size, 1]),
                        [1, 1, self.edim]
                    ),
                ctx),
                1
            ),
            [-1, 1, self.edim]
        )
        # vec = [(batch_size / steps) * steps, 1,  edims]
        return vec