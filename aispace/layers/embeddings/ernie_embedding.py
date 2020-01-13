# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 14:14
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_embedding.py

__all__ = [
    "ErnieEmbedding"
]

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.tf_utils import get_initializer


class ErnieEmbedding(tf.keras.layers.Layer):
    """Construct the embeddings from word, position, token_type and task_type embeddings for bert.
    """
    def __init__(self, hparams: Hparams, **kwargs):
        super(ErnieEmbedding, self).__init__(**kwargs)
        self.vocab_size = hparams.vocab_size
        self.hidden_size = hparams.hidden_size
        self.initializer_range = hparams.initializer_range
        self.use_task_id = hparams.use_task_id

        self.position_embeddings = tf.keras.layers.Embedding(
            hparams.max_position_embeddings,
            hparams.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings"
        )

        self.token_type_embeddings = tf.keras.layers.Embedding(
            hparams.get("type_vocab_size", hparams.get("sent_type_vocab_size")),
            hparams.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings"
        )

        if self.use_task_id:
            self.task_embeddings = tf.keras.layers.Embedding(
                hparams.task_type_vocab_size,
                hparams.hidden_size,
                embeddings_initializer=get_initializer(self.initializer_range),
                name="task_type_embeddings"
            )

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=hparams.layer_norm_eps,
            name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(
            hparams.hidden_dropout_prob
        )

    def build(self, input_shape):
        self.word_embeddings = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range)
        )
        super(ErnieEmbedding, self).build(input_shape)

    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids, task_type_ids = inputs

        seq_length = tf.shape(input_ids)[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(tf.shape(input_ids), 0)

        words_embeddings = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.use_task_id:
            task_type_embeddings = self.task_embeddings(task_type_ids)
            embeddings += task_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])