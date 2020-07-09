# -*- coding: utf-8 -*-
# @Time    : 2020-07-09 10:16
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : qa_with_impossible.py

import tensorflow as tf

from aispace.layers.base_layer import BaseLayer
from aispace.utils.tf_utils import masked_softmax, generate_onehot_label, get_shape
from aispace.layers.activations import ACT2FN

__all__ = [
    'QALayerWithImpossible'
]


@BaseLayer.register("qa_with_impossible")
class QALayerWithImpossible(tf.keras.layers.Layer):
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
        start_result = self.start_project(seq_output)  # [b, l, h] --> [b, l, 1]
        start_result = tf.squeeze(start_result, axis=-1)  # [b, l, 1] --> [b, l]
        start_prob = masked_softmax(start_result, passage_mask)  # [b, l]

        start_top_prob, start_top_index, end_top_prob, end_top_index = [None] * 4
        # end
        if is_training:
            start_index = generate_onehot_label(tf.expand_dims(start_position, axis=-1), self.seq_len)  # [b, 1, l]
            feat_result = tf.matmul(start_index, seq_output)  # [b, 1, l], [b, l, h] --> [b, 1, h]
            feat_result = tf.tile(feat_result, multiples=[1, self.seq_len, 1])  # [b, 1, h] --> [b, l, h]

            end_result = tf.concat([seq_output, feat_result], axis=-1)  # [b, l, h], [b, l, h] --> [b, l, 2h]
            end_result = self.end_modeling(end_result)  # [b, l, 2h] --> [b, l, h]
            end_result = self.layer_norm(end_result)  # [b, l, h] --> [b, l, h]
            end_result = self.end_project(end_result)  # [b, l, h] --> [b, l, 1]
            end_result = tf.squeeze(end_result, axis=-1)  # [b, l, 1] --> [b, l]
            end_prob = masked_softmax(end_result, passage_mask)  # [b, l]
        else:
            start_top_prob, start_top_index = tf.nn.top_k(start_prob, self.start_n_top)  # [b, l] --> [b, k], [b, k]
            start_index = generate_onehot_label(start_top_index, self.seq_len)  # [b, k] --> [b, k, l]
            feat_result = tf.matmul(start_index, seq_output)  # [b, k, l], [b, l, h] -> [b, k, h]
            feat_result = tf.expand_dims(feat_result, axis=1)  # [b, k, h] --> [b, 1, k, h]
            feat_result = tf.tile(feat_result, multiples=[1, self.seq_len, 1, 1])  # [b, 1, k, h] --> [b, l, k, h]

            end_result = tf.expand_dims(seq_output, axis=-2)  # [b, l, h] --> [b, l, 1, h]
            end_result = tf.tile(end_result, multiples=[1, 1, self.start_n_top, 1])  # [b, l, 1, h] --> [b, l, k, h]
            end_result = tf.concat([end_result, feat_result], axis=-1)  # [b, l, k, h], [b, l, k, h] --> [b, l, k, 2h]
            print(get_shape(end_result))
            end_result = self.end_modeling(end_result)  # [b, l, k, 2h] --> [b, l, k, h]
            print(get_shape(end_result))
            end_result = self.layer_norm(end_result)  # [b, l, k, h] --> [b, l, k, h]
            end_result = self.end_project(end_result)  # [b, l, k, h] --> [b, l, k, 1]
            end_result = tf.transpose(tf.squeeze(end_result, axis=-1), perm=[0, 2, 1])  # [b, l, k, 1] --> [b, k, l]

            end_passage_mask = tf.expand_dims(passage_mask, axis=1)  # [b, l] --> [b, 1, l]
            end_passage_mask = tf.tile(end_passage_mask, [1, self.start_n_top, 1])  # [b, 1, l] --> [b, k, l]

            end_prob = masked_softmax(end_result, end_passage_mask)  # [b, k, l]

            end_top_prob, end_top_index = tf.nn.top_k(end_prob, self.start_n_top)  # [b, k, l] --> [b, k, k], [b, k, k]

        # answer
        start_feature = tf.einsum('bl,blh->bh', start_prob, seq_output)  # [b, h]
        answer_result = tf.concat([start_feature, cls_output], axis=-1)  # [b, h], [b, h] --> [b, 2h]
        answer_result = self.answer_modeling(answer_result)  # [b, 2h] --> [b, h]
        answer_result = self.answer_dropout(answer_result)  # [b, h]
        answer_prob = self.answer_project(answer_result)  # [b, h] --> [b, 1]
        answer_prob = tf.squeeze(answer_prob, axis=-1)  # [b, 1] --> [b]
        # answer_prob = tf.sigmoid(answer_result)  # [b]

        if is_training:
            output = (start_prob, end_prob, answer_prob)
        else:
            output = (start_top_prob, start_top_index, end_top_prob, end_top_index, answer_prob)

        return output