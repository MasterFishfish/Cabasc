import tensorflow as tf
from until.Randomer import Randomer
from until.TensorGather import last_relevant

class LSTMlayer():
    # 基本的RNN神经网络模型，前向传播
    def __init__(self, hidden_size,
                 output_keep_prob=0.8,
                 input_keep_prob=1.0,
                 forget_bias=1.0,
                 cell='lstm'):
        # hidden_size: 一个神经网络的隐藏层所含神经元的数目
        # output_keep_prob: 训练过程中输出层的留存率
        # input_keep_prob: 训练过程中输入层的留存率
        # forget_bias: 在lstm,或者GRU中,forget层的bias
        self.hidden_size = hidden_size
        self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        self.forget_bias = forget_bias
        self.first_use = True
        if cell == "lstm":
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=self.hidden_size,
                forget_bias=self.forget_bias,
                state_is_tuple=True
            )
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=self.lstm_cell,
                input_keep_prob=self.input_keep_prob,
                output_keep_prob=self.output_keep_prob
            )
        elif cell == "gru":
            self.lstm_cell = tf.nn.rnn_cell.GRUCell(
                num_units=self.hidden_size
            )
            self.lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=self.lstm_cell,
                input_keep_prob=self.input_keep_prob,
                output_keep_prob=self.output_keep_prob
            )

    def forward(self, inputs, sequence_length, last_outputs=False, init_state=None, name="1"):
        # inputs.shape = [batch_size, steps, edims]
        # inputs为一连串的句子, 为该模型的输入
        # sequence_length 储存inputs里一连串句子的各自长度
        # sequence_length.shape = [batch_size]
        # last_outputs 是否输出lstm最后的时间序列的输出
        # init_state.shape = [batch_size, hidden_size]
        # init_state 展开时间序列时会用到的lstm初始化的值
        # name 用于命名变量名称
        batch_size = tf.shape(inputs)[0]
        if init_state is None:
            lstm_init = self.lstm_cell.zero_state(batch_size=batch_size)
        else:
            lstm_init = tf.Variable(initial_value=init_state)

        # 定义 变量命名域内的变量是否共享
        if self.first_use is True:
            reuse = None
            self.first_use = False
        else:
            reuse = True

        with tf.variable_scope(name, reuse=reuse):
            # outputs 的输出格式为 [batch_size, time_steps, input_size]
            # 因为 major 为 False
            outputs, last_state = tf.nn.dynamic_rnn(
                cell=self.lstm_cell,
                dtype=tf.float32,
                inputs=inputs,
                sequence_length=sequence_length,
                init_state=lstm_init,
                time_major=False
            )
            if last_outputs:
                # 如果输出 last_outputs
                # outputs.shape = [batch_size, num_units]
                outputs = last_relevant(inputs=inputs,
                                        sequence_length=sequence_length)
                outputs = tf.reshape(outputs, shape=[batch_size, 1, self.hidden_size])
            return outputs, last_state
