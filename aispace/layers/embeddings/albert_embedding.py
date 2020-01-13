# -*- coding: utf-8 -*-
# @Time    : 2019-12-01 14:33
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : albert_embedding.py

__all__ = [
    "AlbertEmbedding",
    "AlbertEmbeddingV2"
]

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.tf_utils import get_initializer


class AlbertEmbedding(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings for albert.
    google or huggingface version
    """
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertEmbedding, self).__init__(**kwargs)
        self.max_position_embeddings = hparams.max_position_embeddings
        self.embedding_size = hparams.embedding_size
        self.initializer_range = hparams.initializer_range
        self.layer_norm_eps = hparams.layer_norm_eps
        self.vocab_size = hparams.vocab_size
        self.type_vocab_size = hparams.type_vocab_size
        self.hidden_dropout_prob = hparams.hidden_dropout_prob

        self.position_embeddings = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name='position_embeddings')
        self.token_type_embeddings = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.embedding_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name='token_type_embeddings')

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range))
        super(AlbertEmbedding, self).build(input_shape)

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
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, embedding_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        x = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)
        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class AlbertEmbeddingV2(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings for albert.
    brightmart version
    """
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertEmbeddingV2, self).__init__(**kwargs)
        self.max_position_embeddings = hparams.max_position_embeddings
        self.hidden_size = hparams.hidden_size
        self.embedding_size = hparams.embedding_size
        self.initializer_range = hparams.initializer_range
        self.layer_norm_eps = hparams.layer_norm_eps
        self.vocab_size = hparams.vocab_size
        self.type_vocab_size = hparams.type_vocab_size
        self.hidden_dropout_prob = hparams.hidden_dropout_prob

        # self.embedding_hidden_mapping_in = tf.keras.layers.Dense(
        #     self.hidden_size,
        #     kernel_initializer=get_initializer(self.initializer_range),
        #     name='embedding_hidden_mapping_in'
        # )
        self.position_embeddings = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name='position_embeddings')
        self.token_type_embeddings = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name='token_type_embeddings')

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range))
        self.embedding_hidden_mapping_in = self.add_weight(
            "embedding_hidden_mapping_in",
            shape=[self.embedding_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range)
        )
        super(AlbertEmbeddingV2, self).build(input_shape)

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
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        inputs_embeds = tf.matmul(inputs_embeds, self.embedding_hidden_mapping_in) # self.embedding_hidden_mapping_in(inputs_embeds)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, embedding_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        x = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)
        return tf.reshape(logits, [batch_size, length, self.vocab_size])