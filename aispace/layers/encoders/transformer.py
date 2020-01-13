# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 16:07
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : transformer.py

__all__ = [
    "Transformer",
    "TransformerAttention",
    "TransformerBlock",
    "TransformerIntermediate"
]

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.layers.attentions.multi_head_attention import MultiHeadAttention
from aispace.layers.fusions.feed_forward_add_and_norm import FeedForwardAddAndNorm
from aispace.utils.tf_utils import get_initializer
from aispace.layers.activations import ACT2FN


class TransformerAttention(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(TransformerAttention, self).__init__(**kwargs)

        self.self_attention = MultiHeadAttention(hparams, name="self")
        self.dense_output = FeedForwardAddAndNorm(hparams, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask = inputs

        self_outputs = self.self_attention(
            input_tensor, input_tensor, input_tensor,
            attention_mask=attention_mask, head_mask=head_mask, training=training)
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TransformerIntermediate(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(TransformerIntermediate, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hparams.intermediate_size,
            kernel_initializer=get_initializer(hparams.initializer_range),
            name='dense'
        )
        self.intermediate_act_fn = ACT2FN[hparams.hidden_act]

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = TransformerAttention(hparams, name="attention")
        self.intermediate = TransformerIntermediate(hparams, name="intermediate")
        self.bert_output = FeedForwardAddAndNorm(hparams, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        attention_outputs = self.attention([hidden_states, attention_mask, head_mask], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class Transformer(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.output_attentions = hparams.output_attentions
        self.output_hidden_states = hparams.output_hidden_states
        self.layers = [TransformerBlock(hparams, name=f"layer_._{i}") for i in range(hparams.num_hidden_layers)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module([hidden_states, attention_mask, head_mask[i]], training=training)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)