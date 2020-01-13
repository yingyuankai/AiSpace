# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 14:59
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : self_attention.py

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.tf_utils import get_initializer, get_shape


__all__ = [
    "MultiHeadAttention"
]


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        if hparams.hidden_size % hparams.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hparams.hidden_size} is not a multiple of the number of attention "
                f"heads {hparams.num_attention_heads}")
        self.output_attentions = hparams.output_attentions

        self.num_attention_heads = hparams.num_attention_heads
        assert hparams.hidden_size % hparams.num_attention_heads == 0
        self.attention_head_size = int(hparams.hidden_size / hparams.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(hparams.initializer_range),
            name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(hparams.initializer_range),
            name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(hparams.initializer_range),
            name="value"
        )
        self.dropout = tf.keras.layers.Dropout(
            hparams.attention_probs_dropout_prob
        )

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, attention_mask=None, head_mask=None, training=False):
        batch_size = get_shape(query)[0]

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer,
                                     transpose_b=True)  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(get_shape(key_layer)[-1], tf.float32)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,
                                   (batch_size, -1, self.all_head_size))  # (batch_size, seq_len_q, all_head_size)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs