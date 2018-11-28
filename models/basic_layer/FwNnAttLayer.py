# 该类定义了论文中的BaseA模型
import tensorflow as tf
from Linearlayer_3dim import Linearlayer_3dim
from until.Activer import activer
from until.Randomer import Randomer
from until.SoftMask import normalizer

class FwNnAttLayer():
    def __init__(self, edims, active="tanh", stddev=None, params=None, norm_type=None):
        self.edims = edims
        self.stddev = stddev
        self.active = active
        self.norm_type = norm_type
        if params is None:
            w_ctx = None
            w_asp = None
            w_att_ca = None
        else:
            w_ctx = params["wline_ctx"]
            w_asp = params["wline_asp"]
            w_att_ca = params["wline_att_ca"]
        self.w_linear_ctx = Linearlayer_3dim(
            w_shape=[self.edims, self.edims],
            stddev=self.stddev, param=w_ctx
        )
        self.w_linear_asp = Linearlayer_3dim(
            w_shape=[self.edims, self.edims],
            stddev=self.stddev, param=w_asp
        )
        self.weights_attention = w_att_ca or tf.Variable(
            initial_value=Randomer.random_normal(wshape=[self.edims, 1], stddev=stddev),
            trainable=True
        )
    def count_alpha(self, contexts, aspects, ctx_bitmap, alpha_adj=None):
        time_steps = tf.shape(contexts)[1]
        ctx = contexts
        asp = aspects
        asp_3dims = tf.reshape(
            tensor=tf.tile(asp, [1, time_steps]),
            shape=[-1, time_steps, self.edims]
        )

        res_ctx = self.w_linear_ctx.forward(inputs=ctx)
        res_asp = self.w_linear_asp.forward(inputs=asp_3dims)
        res = res_ctx + res_asp
        res_act = activer(vec=res, active=self.active)

        batch_size = tf.shape(ctx)[0]
        watt_0 = tf.shape(self.weights_attention)[0]
        watt_1 = tf.shape(self.weights_attention)[1]
        w_att_3dims = tf.reshape(
            tensor=tf.tile(self.weights_attention, [batch_size, 1]),
            shape=[-1, watt_0, watt_1]
        )
        # 一定要化成二维的矩阵
        vec = tf.reshape(
            tensor=tf.matmul(res_act, w_att_3dims),
            shape=[-1, time_steps]
        )
        alpha = normalizer(norm_type=self.norm_type, inputs=vec, seq_mask=ctx_bitmap, axis=1)
        if alpha_adj is not None:
            alpha += alpha_adj

        return alpha


    def forward(self, contexts, aspects, ctx_bitmap, alpha_adj=None):
        mem_size = tf.shape(contexts)[1]
        ctx = contexts
        asp = aspects
        alpha = self.count_alpha(contexts=ctx, aspects=asp, ctx_bitmap=ctx_bitmap, alpha_adj=alpha_adj)
        alpha_3dims = tf.reshape(
            tensor=alpha, shape=[-1, 1, mem_size]
        )

        sentence_embedding = tf.matmul(alpha_3dims, ctx)
        return sentence_embedding, alpha_3dims

    def forward_wot_sum(self, contexts, aspects, ctx_bitmap, alpha_adj=None):
        mem_size = tf.shape(contexts)[1]
        ctx = contexts
        asp = aspects
        alpha = self.count_alpha(contexts=ctx, aspects=asp, ctx_bitmap=ctx_bitmap, alpha_adj=alpha_adj)
        alpha_shaped = tf.reshape(
            tensor=alpha, shape=[-1, mem_size, 1]
        )
        alpha_titled = tf.reshape(
            tensor=tf.tile(alpha_shaped, [1, 1, self.edims])
        )
        ret = ctx * alpha_titled
        return ret, alpha