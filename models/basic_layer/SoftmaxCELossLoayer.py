import tensorflow as tf
from until.Randomer import Randomer
from Linearlayer_3dim import Linearlayer_3dim

class SoftmaxCELoassLayer():
    def __init__(self, edims, num_class, stddev=None, params=None):
        self.edims = edims
        self.num_class = num_class
        self.stddev = stddev
        if params is None:
            self.w = tf.Variable(
                initial_value=Randomer.random_normal(shape=[self.edims, self.num_class], stddev=self.stddev),
                trainable=True
            )
            self.b = tf.Variable(
                initial_value=tf.zeros([1, 1]),
                trainable=True
            )
        # params = {"wline" wline, "bline": bline}
        else:
            self.w = params["wline"]
            self.b = params["bline"]

    def forward(self, inputs):
        # inputs.shape = [batch_size, edims]
        softmax_input = tf.add(tf.matmul(inputs, self.edims), self.b)
        # pred.shape = [batch_size, 1]
        pred = tf.argmax(tf.nn.softmax(softmax_input), axis=1)
        return pred, softmax_input

    def get_loss(self, softmax_input, res_labels):
        # labels = [batch_size, num_class]
        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_input, labels=res_labels)
        return loss



