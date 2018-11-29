import tensorflow as tf
from until.Randomer import Randomer
from until.SoftMask import softmax_mask
from Linearlayer_3dim import Linearlayer_3dim
from until.SoftMask import normalizer
class ConcatAttention():
    def __init__(self, edim, stddev=None, norm_type="softmax"):
        self.edim = edim
        self.stddev = stddev
        self.w_ca = Linearlayer_3dim(
            w_shape=[self.edim * 2, 1],
            stddev=stddev
        )
        self.norm_type = norm_type

    def count_alpha(self, contexts, aspects, ctx_bitmap):
        time_steps = tf.shape(contexts)[1]
        batch_size = tf.shape(contexts)[0]
        ctx = contexts
        asp = aspects
        asp_shaped = tf.reshape(asp, [-1, 1, self.edim])
        asp_tiled = tf.tile(asp_shaped, [1, time_steps, self.edim])
        rec_ct_ca = tf.concat([ctx, asp_tiled], axis=2)
        res = self.w_ca.forward(inputs=rec_ct_ca)
        res = tf.reshape(res, [-1, time_steps])
        alpha = normalizer(norm_type=self.norm_type, inputs=res, seq_mask=ctx_bitmap)
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
        # vec.shape = [batch_size, 1, edims]
        return vec, alpha

    def forward_max_pools(self, context, aspect, ctx_bitmap):
        mem_size = tf.shape(context)[1]
        ctx = context
        asp = aspect
        # adjust attention
        alpha = self.count_alpha(ctx, asp, ctx_bitmap)
        vec = tf.reshape(
            tf.reduce_max(
                tf.multiply(
                    tf.tile(
                        tf.reshape(alpha, [-1, mem_size, 1]),
                        [1, 1, self.edim]
                    ),
                    ctx
                ),
                1
            ),
            [-1, 1, self.edim]
        )
        return vec