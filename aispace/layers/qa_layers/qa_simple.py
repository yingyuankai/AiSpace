# -*- coding: utf-8 -*-
# @Time    : 2020-07-09 10:16
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : qa_with_impossible.py

import tensorflow as tf

from aispace.layers.base_layer import BaseLayer
from aispace.utils.tf_utils import masked_softmax, generate_onehot_label, mask_logits
from aispace.layers.activations import ACT2FN

__all__ = [
    'QALayerSimple'
]


@BaseLayer.register("qa_simple")
class QALayerSimple(tf.keras.layers.Layer):
    """
    QA Layer just simple!
    """
    def __init__(self, hidden_size, seq_len, start_n_top, end_n_top, initializer, dropout, layer_norm_eps=1e-12, **kwargs):
        super(QALayerSimple, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, unused_input_shapes):
        # for start
        self.start_project = tf.keras.layers.Dense(
            1,
            kernel_initializer=self.initializer,
            name="start_project"
        )

        # for end
        self.end_project = tf.keras.layers.Dense(
            1,
            kernel_initializer=self.initializer,
            name="end_project"
        )
        super(QALayerSimple, self).build(unused_input_shapes)

    def call(self, inputs, **kwargs):
        seq_output, cls_output, passage_mask, start_position = inputs
        is_training = kwargs.get("training", False)

        start_feature = self.start_project(seq_output)  # [b, l, h] --> [b, l, 1]
        start_feature = tf.squeeze(start_feature, axis=-1)  # [b, l, 1] --> [b, l]
        start_log_probs = masked_softmax(start_feature, passage_mask, is_training)  # [b, l]

        end_feature = self.end_project(seq_output)  # [b, l, h] --> [b, l, 1]
        end_feature = tf.squeeze(end_feature, axis=-1)  # [b, l, 1] --> [b, l]
        end_log_probs = masked_softmax(end_feature, passage_mask, is_training)  # [b, l]

        if is_training:
            output = (start_log_probs, end_log_probs)
        else:
            start_top_log_prob, start_top_index = tf.nn.top_k(start_log_probs,
                                                              self.start_n_top)  # [b, l] --> [b, k], [b, k]
            end_top_log_prob, end_top_index = tf.nn.top_k(end_log_probs,
                                                          self.start_n_top)  # [b, k, l] --> [b, k], [b, k]

            start_top_log_prob = tf.expand_dims(start_top_log_prob, axis=-1)
            start_top_index = tf.expand_dims(tf.cast(start_top_index, dtype=tf.float32), axis=-1)
            start_top_res = tf.concat([start_top_log_prob, start_top_index], axis=-1)

            end_top_log_prob = tf.expand_dims(end_top_log_prob, axis=-1)
            end_top_index = tf.expand_dims(tf.cast(end_top_index, dtype=tf.float32), axis=-1)
            end_top_res = tf.concat([end_top_log_prob, end_top_index], axis=-1)

            output = (start_top_res, end_top_res)

        return output