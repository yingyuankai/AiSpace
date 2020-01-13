# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 15:44
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : feed_forward_add_and_norm.py


import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.tf_utils import get_initializer


class FeedForwardAddAndNorm(tf.keras.layers.Layer):
    """Ref to Bert feed forward and add & norm module"""
    def __init__(self, hparams: Hparams, **kwargs):
        super(FeedForwardAddAndNorm, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            hparams.hidden_size,
            kernel_initializer=get_initializer(hparams.initializer_range),
            name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=hparams.layer_norm_eps,
            name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(
            hparams.hidden_dropout_prob
        )

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states