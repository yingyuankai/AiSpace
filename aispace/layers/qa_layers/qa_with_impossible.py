# -*- coding: utf-8 -*-
# @Time    : 2020-07-09 10:16
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : qa_with_impossible.py

import tensorflow as tf

from aispace.layers.base_layer import BaseLayer
from aispace.utils.tf_utils import mask_logits, generate_onehot_label
from aispace.layers.activations import ACT2FN

__all__ = [
    'QALayerWithImpossible'
]


@BaseLayer.register("qa_with_impossible")
class QALayerWithImpossible(tf.keras.layers.Layer):
    """
    refs:
    https://github.com/kamalkraj/ALBERT-TF2.0/blob/master/run_squad.py
    https://github.com/stevezheng23/xlnet_extension_tf/blob/master/run_squad.py#L597
    """
    def __init__(self, hidden_size, seq_len, start_n_top, end_n_top, initializer, dropout, layer_norm_eps=1e-12, **kwargs):
        super(QALayerWithImpossible, self).__init__(**kwargs)
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
        self.end_modeling = tf.keras.layers.Dense(
            self.hidden_size,
            activation=ACT2FN['tanh'],
            kernel_initializer=self.initializer,
            name="end_modeling"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=self.layer_norm_eps,
            name="layer_norm"
        )
        self.end_project = tf.keras.layers.Dense(
            1,
            kernel_initializer=self.initializer,
            name="end_project"
        )

        # for answer
        self.answer_modeling = tf.keras.layers.Dense(
            self.hidden_size,
            activation=ACT2FN['tanh'],
            kernel_initializer=self.initializer,
            name="answer_modeling"
        )
        self.answer_dropout = tf.keras.layers.Dropout(self.dropout)
        self.answer_project = tf.keras.layers.Dense(
            1,
            kernel_initializer=self.initializer,
            name="answer_project"
        )
        super(QALayerWithImpossible, self).build(unused_input_shapes)

    def call(self, inputs, **kwargs):
        seq_output, cls_output, passage_mask, start_position = inputs
        is_training = kwargs.get("training", False)

        start_feature = self.start_project(seq_output)  # [b, l, h] --> [b, l, 1]
        start_feature = tf.squeeze(start_feature, axis=-1)  # [b, l, 1] --> [b, l]
        start_logits_masked = mask_logits(start_feature, passage_mask)  # [b, l]
        start_log_probs = tf.nn.log_softmax(start_logits_masked, axis=-1)

        start_top_log_prob, start_top_index, end_top_log_prob, end_top_index = [None] * 4
        # end
        if is_training:
            start_index = generate_onehot_label(start_position, self.seq_len)  # [b, 1, l]

            start_feature = tf.matmul(start_index, seq_output)  # [b, 1, l], [b, l, h] --> [b, 1, h]
            start_feature = tf.tile(start_feature, multiples=[1, self.seq_len, 1])  # [b, 1, h] --> [b, l, h]

            end_feature = tf.concat([seq_output, start_feature], axis=-1)  # [b, l, h], [b, l, h] --> [b, l, 2h]
            end_feature = self.end_modeling(end_feature)  # [b, l, 2h] --> [b, l, h]
            end_feature = self.layer_norm(end_feature)  # [b, l, h] --> [b, l, h]
            end_feature = self.end_project(end_feature)  # [b, l, h] --> [b, l, 1]
            end_feature = tf.squeeze(end_feature, axis=-1)  # [b, l, 1] --> [b, l]
            end_logits_masked = mask_logits(end_feature, passage_mask)  # [b, l]
            end_log_probs = tf.nn.log_softmax(end_logits_masked, axis=-1)
        else:
            start_top_log_prob, start_top_index = tf.nn.top_k(start_log_probs, self.start_n_top)  # [b, l] --> [b, k], [b, k]
            start_index = generate_onehot_label(start_top_index, self.seq_len)  # [b, k] --> [b, k, l]

            start_feature = tf.matmul(start_index, seq_output)  # [b, k, l], [b, l, h] -> [b, k, h]
            start_feature = tf.expand_dims(start_feature, axis=1)  # [b, k, h] --> [b, 1, k, h]
            start_feature = tf.tile(start_feature, multiples=[1, self.seq_len, 1, 1])  # [b, 1, k, h] --> [b, l, k, h]

            end_feature = tf.expand_dims(seq_output, axis=-2)  # [b, l, h] --> [b, l, 1, h]
            end_feature = tf.tile(end_feature, multiples=[1, 1, self.start_n_top, 1])  # [b, l, 1, h] --> [b, l, k, h]
            end_feature = tf.concat([end_feature, start_feature], axis=-1)  # [b, l, k, h], [b, l, k, h] --> [b, l, k, 2h]
            end_feature = self.end_modeling(end_feature)  # [b, l, k, 2h] --> [b, l, k, h]
            end_feature = tf.reshape(end_feature, [-1, self.seq_len * self.start_n_top, self.hidden_size])   # [b, l, k, h] --> [b, lk, h]
            end_feature = self.layer_norm(end_feature)  # [b, lk, h] --> [b, lk, h]
            end_feature = tf.reshape(end_feature, [-1, self.seq_len, self.start_n_top, self.hidden_size])  # [b, lk, h] --> [b, l, k, h]
            end_feature = self.end_project(end_feature)  # [b, l, k, h] --> [b, l, k, 1]
            end_feature = tf.transpose(tf.squeeze(end_feature, axis=-1), perm=[0, 2, 1])  # [b, l, k, 1] --> [b, k, l]

            end_passage_mask = tf.expand_dims(passage_mask, axis=1)  # [b, l] --> [b, 1, l]
            end_passage_mask = tf.tile(end_passage_mask, [1, self.start_n_top, 1])  # [b, 1, l] --> [b, k, l]

            end_logits_masked = mask_logits(end_feature, end_passage_mask)  # [b, k, l]
            end_log_probs = tf.nn.log_softmax(end_logits_masked, axis=-1)

            end_top_log_prob, end_top_index = tf.nn.top_k(end_log_probs, self.start_n_top)  # [b, k, l] --> [b, k, k], [b, k, k]

        # answer
        start_feature = tf.nn.softmax(start_logits_masked, axis=-1, name="softmax_start")
        start_feature = tf.einsum('bl,blh->bh', start_feature, seq_output)  # [b, h]

        answer_feature = tf.concat([cls_output, start_feature], axis=-1)  # [b, h], [b, h] --> [b, 2h]
        answer_feature = self.answer_modeling(answer_feature)  # [b, 2h] --> [b, h]
        answer_feature = self.answer_dropout(answer_feature)  # [b, h]
        answer_prob = self.answer_project(answer_feature)  # [b, h] --> [b, 1]
        # answer_prob = tf.squeeze(answer_prob, axis=-1)  # [b, 1] --> [b]
        # answer_prob = tf.sigmoid(answer_result)  # [b]

        if is_training:
            output = (start_log_probs, end_log_probs, answer_prob)
        else:
            output = (start_top_log_prob, start_top_index, end_top_log_prob, end_top_index, answer_prob)

        return output