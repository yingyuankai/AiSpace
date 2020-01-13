# -*- coding: utf-8 -*-
# @Time    : 2019-11-29 11:19
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : rnn.py

import tensorflow as tf

__all__ = [
    "Bilstm"
]


class Bilstm(tf.keras.layers.Layer):
    def __init__(self, units, dropout, **kwargs):
        super(Bilstm, self).__init__(**kwargs)
        fwd_lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            go_backwards=False,
            dropout=dropout,
            name="fwd_lstm")
        bwd_lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            go_backwards=True,
            dropout=dropout,
            name="bwd_lstm")
        self.bilstm = tf.keras.layers.Bidirectional(
            merge_mode="concat",
            layer=fwd_lstm,
            backward_layer=bwd_lstm,
            name="bilstm")

    def call(self, inputs, **kwargs):
        outputs = self.bilstm(inputs, training=kwargs.get('training', False))
        return outputs