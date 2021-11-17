# -*- coding: utf-8 -*-
# @Time    : 2019-12-01 18:12
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : sequence_pooler.py

__all__ = [
    "SequenceSummary"
]

import tensorflow as tf

from aispace.utils.tf_utils import get_initializer, get_shape
from aispace.utils.hparams import Hparams


class SequenceSummary(tf.keras.layers.Layer):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """
    def __init__(self, config: Hparams, **kwargs):
        super(SequenceSummary, self).__init__(**kwargs)
        initializer_range = config.initializer_range

        self.summary_type = config.summary_type if 'summary_use_proj' in config else 'last'
        if self.summary_type == 'attn':
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.has_summary = 'summary_use_proj' in config and config.summary_use_proj
        if self.has_summary:
            if 'summary_proj_to_labels' in config and "num_labels" in config and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=get_initializer(initializer_range),
                name='summary')

        self.has_activation = 'summary_activation' in config and config.summary_activation == 'tanh'
        if self.has_activation:
            self.activation = tf.keras.activations.tanh

        self.has_first_dropout = 'summary_first_dropout' in config and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = tf.keras.layers.Dropout(config.summary_first_dropout)

        self.has_last_dropout = 'summary_last_dropout' in config and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = tf.keras.layers.Dropout(config.summary_last_dropout)

    def call(self, inputs, training=False):
        """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
            cls_index = None
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."
        else:
            input_ids = inputs.get('input_ids')
            cls_index = inputs.get('cls_index', None)

        if self.summary_type == 'last':
            output = hidden_states[:, -1]
        elif self.summary_type == 'first':
            output = hidden_states[:, 0]
        elif self.summary_type == 'mean':
            output = tf.mean(hidden_states, axis=1)
        elif self.summary_type == 'cls_index':
            hidden_shape = get_shape(hidden_states)  # e.g. [batch, num choices, seq length, hidden dims]
            if cls_index is None:
                cls_index = tf.fill(hidden_shape[:-2], hidden_shape[-2] - 1)  # A tensor full of shape [batch] or [batch, num choices] full of sequence length
            cls_shape = get_shape(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = cls_index[..., tf.newaxis]
            # else:
                # cls_index = cls_index[..., tf.newaxis]
                # cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(output, axis=len(hidden_shape) - 2) # shape of output: (batch, num choices, hidden_size)
        elif self.summary_type == 'attn':
            raise NotImplementedError

        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        if self.has_summary:
            output = self.summary(output)

        if self.has_activation:
            output = self.activation(output)

        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        return output