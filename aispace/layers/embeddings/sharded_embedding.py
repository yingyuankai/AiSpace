# -*- coding: utf-8 -*-
# @Time    : 2019-12-01 17:23
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : sharded_embedding.py

__all__ = [
    "SharedEmbeddings"
]

import tensorflow as tf

from aispace.utils.tf_utils import get_initializer, get_shape


class SharedEmbeddings(tf.keras.layers.Layer):
    """Construct shared token embeddings.
    """

    def __init__(self, vocab_size, hidden_size, initializer_range=None, **kwargs):
        super(SharedEmbeddings, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size ** -0.5 if initializer_range is None else initializer_range

    def build(self, input_shape):
        """Build shared word embedding layer
        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.hidden_size],
            initializer=get_initializer(self.initializer_range))
        super(SharedEmbeddings, self).build(input_shape)

    def call(self, inputs, mode="embedding"):
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
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [..., hidden_size]
            Returns:
                float32 tensor with shape [..., vocab_size].
        """
        first_dims = get_shape(inputs)[:-1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])