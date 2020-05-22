# -*- coding: utf-8 -*-
# @Time    : 2019-11-15 11:15
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : crf.py


import tensorflow as tf
import tensorflow_addons as tfa

from aispace.utils.tf_utils import get_initializer, get_shape


__all__ = [
    "CRFLayer"
]


class CRFLayer(tf.keras.layers.Layer):
    def __init__(self,
                 num_labels,
                 initializer_range,
                 label_mask=None,
                 **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        if label_mask is not None:
            self.label_mask = tf.constant(label_mask)
        else:
            self.label_mask = None

    def build(self, input_shape):
        self.transition_params = self.add_weight(
            "transition_params",
            shape=[self.num_labels, self.num_labels],
            initializer=get_initializer(self.initializer_range)
        )
        if self.label_mask is not None:
            label_mask = tf.cast(self.label_mask, tf.float32)
            label_mask = (1.0 - label_mask) * -10000.0
            self.transition_params += label_mask

        super(CRFLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        score, self.sequence_length = inputs
        viterbi, viterbi_score = tfa.text.crf_decode(score, self.transition_params, self.sequence_length)
        return viterbi, viterbi_score

    def loss(self, y_true, y_pred):
        """Computes the log-likelihood of tag sequences in a CRF.
        Args:
            y_true : A (batch_size, n_steps) tensor.
            y_pred : A (batch_size, n_steps, n_classes) tensor.
        Returns:
            loss: A scalar containing the log-likelihood of the given sequence of tag indices.
        """
        batch_size, n_steps, _ = get_shape(y_pred)
        y_true = tf.cast(tf.reshape(y_true, [batch_size, n_steps]), dtype='int32')
        log_likelihood, self.transition_params = \
            tfa.text.crf_log_likelihood(y_pred, y_true, self.sequence_length, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)
        return loss