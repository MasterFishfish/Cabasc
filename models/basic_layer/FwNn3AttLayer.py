# 该类定义了论文中的BaseB的模型
import tensorflow as tf
import numpy as np
from Linearlayer_3dim import Linearlayer_3dim
from until.Randomer import Randomer
from until.Activer import activer
from until.SoftMask import normalizer

class FwNnAttLayer():
    def __init__(self,
                 edim,
                 active='tanh',
                 stddev=None,
                 param=None,
                 norm_type='softmax'
                 ):
        # edim(int): 为嵌入词向量的维度
        # active(string): 为隐层的激励函数
        # stddev(float): 创建随机矩阵的参数
        # param(dict): param["wline_ctx"] 传入模型输入层的context weights的初始参数
        #              param["wline_asp"] 传入模型输入层的aspect weights的初始参数
        #              param["wline_out"] 用于平滑每一个word对于aspect sentiment影响的向量(vs)的weights
        #              param["wline_att_ca"] 模型中的每一段时间序列经过前向传播之后的输出层weights参数，
        #                                    此处的输出层为softmax的输入层
        # 注意param["wline_ctx"] = {"wline": init_weights}
        #     param["wline_asp"] = {"wline": init_weights}
        #     param["wline_output"] = {"wline": init_weights}
        #     param["wline_att_ca"] = init_weights(tensor)  ----  init_weights.shape=[input_edims, 1]
        self.active = active
        self.stddev = stddev
        self.edim = edim
        self.norm_type = norm_type
        if param == None:
            wline_asp = None
            wline_ctx = None
            wline_out = None
            wline_att_ca = None
        else:
            wline_asp = param["wline_asp"]
            wline_ctx = param["wline_ctx"]
            wline_out = param["wline_out"]
            wline_att_ca = param["wline_att_ca"]

        self.w_linear_ctx = Linearlayer_3dim(
            [self.edim, self.edim],
            stddev=self.stddev, param=wline_ctx
        )
        self.w_linear_asp = Linearlayer_3dim(
            [self.edim, self.edim],
            stddev=self.stddev, param=wline_asp
        )
        self.w_linear_out = Linearlayer_3dim(
            [self.edim, self.edim],
            stddev=self.stddev, param=wline_out
        )
        self.w_att_ca = wline_att_ca or tf.Variable(
            initial_value=Randomer.random_normal(wshape=[self.edims, 1], stddev=self.stddev),
            trainable=True
        )

    def count_alpha(self, aspects, context, outputs, ctx_bitmap, alpha_adj=None):
        # aspect.shape = [batch_size, edims]
        # context.shape = [batch_size, steps, edims]
        # output.shape = [batch_size, edims]
        # ctx_bitmap 一个mask，整个模型存在池化操作，为了移除padding的影响
        # 将输入的aspect, 扩展到steps个时间序列
        # 将输入的output, 扩展到steps个时间序列
        # 将初始化的类变量self.w_att_ca, 扩展到batch_size个

        batch_size = tf.shape(context)[0]
        steps = tf.shape(context)[1]
        asp = tf.reshape(
            tensor=tf.tile(aspects, [1, steps]),
            shape=[-1, steps, self.edim]
        )
        out = tf.reshape(
            tensor=tf.tile(outputs, [1, steps]),
            shape=[-1, steps, self.edim]
        )

        res_asp = self.w_linear_asp.forward(inputs=asp)
        res_ctx = self.w_linear_ctx.forward(inputs=context)
        res_out = self.w_linear_out.forward(inputs=out)

        # res.shape = [batch_size, steps, edims]
        res = res_asp + res_ctx + res_out

        # weight_att_ca_3dim.shape = [batch_size, edims, 1]
        weight_att_ca_3dim = tf.reshape(
            tensor=tf.tile(self.w_att_ca, [batch_size, 1]),
            shape=[-1, self.edim, 1]
        )
        res_act = activer(res, self.active)

        # tf.matmul(res_act, weight_att_ca_3dim).shape = [batch_size, steps, 1]
        # softmax_input.shape = [batch_size, steps]
        softmax_input = tf.reshape(
            tensor=tf.matmul(res_act, weight_att_ca_3dim),
            shape=[-1, steps]
        )

        # 从softmax_input中计算出softmax的alpha值
        # 此时softmax_alpha.shape = [batch_size, steps, 1]
        alpha = normalizer(self.norm_type, softmax_input, ctx_bitmap, 1)

        # alpha_adj使用的理由不明
        if alpha_adj is not None:
            alpha += alpha_adj

        return alpha

    def count_alpha2(self, aspects, context, outputs, ctx_bitmap, alpha_adj=None):
        # context.shape = [batch_size, steps, edims]
        # aspect.shape = [batch_size, edims]
        # output = [batch_size, steps, edims]
        # ctx_bitmap.shape = context.shape
        time_steps = tf.shape(context)[1]
        asp = tf.reshape(
            tensor = tf.tile(aspects, [time_steps, 1]),
            shape=[-1, time_steps, self.edim]
        )

        res_asp = self.w_linear_asp.forward(inputs=asp)
        res_ctx = self.w_linear_ctx.forward(inputs=context)
        res_out = self.w_linear_out.forward(inputs=outputs)

        res = res_asp + res_ctx + res_out
        res_act = activer(res, self.active)

        batch_size = tf.shape(context)[0]
        w_0 = tf.shape(self.w_att_ca)[0]
        w_1 = tf.shape(self.w_att_ca)[1]
        w_att_ca_3dims = tf.reshape(
            tensor=tf.tile(self.w_att_ca, [batch_size, 1]),
            shape=[batch_size, w_0, w_1]
        )

        # rec_act.shape = [batch_size, steps, 1]
        rec_act = tf.reshape(
            tensor=tf.matmul(res_act, self.w_att_ca_3dims),
            shape=[-1, time_steps]
        )

        alpha = normalizer(norm_type=self.norm_type, inputs=rec_act, seq_mask=ctx_bitmap, axis=1)

        if alpha_adj is not None:
            alpha += alpha_adj
        return alpha

    def forward(self, context, aspect, output, ctx_bitmap, alpha_adj=None):
        mem_size = tf.shape(context)[1]
        ctx = context
        asp = aspect
        out = output
        alpha = self.count_alpha(aspects=asp,
                                 context=ctx,
                                 outputs=out,
                                 ctx_bitmap=ctx_bitmap,
                                 alpha_adj=alpha_adj)
        alpha_3dims = tf.reshape(
            tensor=alpha, shape=[-1, 1, mem_size]
        )

        sentence_embedding = tf.matmul(alpha_3dims, ctx)
        return sentence_embedding, alpha_3dims

    def forward2(self, context, aspect, output, ctx_bitmap, alpha_adj=None):
        mem_size = tf.shape(context)[1]
        ctx = context
        asp = aspect
        out = output
        alpha = self.count_alpha2(aspects=asp, context=ctx, outputs=out,
                                  ctx_bitmap=ctx_bitmap, alpha_adj=alpha_adj)
        alpha_3dims = tf.reshape(
            tensor=alpha, shape=[-1, 1, mem_size]
        )
        sentence_embedding = tf.matmul(alpha_3dims, ctx)
        return sentence_embedding, alpha_3dims

    def forward_p(self, context, aspect, output, ctx_bitmap, location, alpha_adj=None):
        # 似乎限制了输入context的location 至今不明确用意
        # location.shape = [batch_size, steps]
        mem_size = tf.shape(context)[1]
        ctx = context
        asp = aspect
        out = output
        alpha = self.count_alpha(aspects=asp, outputs=out, ctx_bitmap=ctx_bitmap,
                                 alpha_adj=alpha_adj, context=ctx)
        sentence_embedding = tf.matmul(
            tf.add(tf.reshape(alpha, [-1, 1, mem_size]), location),
            ctx
        )
        return sentence_embedding, alpha

    def forward_wot_sum(self, context, aspect, output, ctx_bitmap, alpha_adj=None):
        # 该函数返回的是根据attention计算出的sentence_embedding
        # mem_size 即context的steps
        mem_size = tf.shape(context)[1]
        ctx = context
        asp = aspect
        out = output
        # alpha.shape = [batch_size, steps]
        alpha = self.count_alpha(aspects=asp, outputs=out, ctx_bitmap=ctx_bitmap,
                                 alpha_adj=alpha_adj, context=ctx)
        # alpha_shaped.shape = [batch_size, steps, 1]
        alpha_shaped = tf.reshape(
            tensor=alpha,
            shape=[-1, mem_size, 1]
        )
        # alpha_tiled.shape = [batch_size, steps, edims]
        alpha_tiled = tf.tile(
            alpha_shaped,
            [1, 1, self.edim]
        )
        ctx_adjed =  ctx * alpha_tiled
        return ctx_adjed, alpha