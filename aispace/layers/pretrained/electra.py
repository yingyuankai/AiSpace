# -*- coding: utf-8 -*-
# @Time    : 2020-06-03 11:34
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : electra.py

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.layers.encoders.transformer import Transformer
from aispace.layers.pretrained.bert import BertPooler
from aispace.layers.embeddings import ElectraEmbeddings
from aispace.layers.base_layer import BaseLayer
from aispace.utils.tf_utils import get_shape
from aispace.layers.activations import ACT2FN


class TFElectraDiscriminatorPredictions(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(hparams.hidden_size, name="dense")
        self.dense_prediction = tf.keras.layers.Dense(1, name="dense_prediction")
        self._act_fn = ACT2FN[hparams.hidden_act]

    def call(self, discriminator_hidden_states, training=False):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self._act_fn(hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states))

        return logits


class TFElectraGeneratorPredictions(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super().__init__(**kwargs)

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=hparams.layer_norm_eps, name="LayerNorm")
        self.dense = tf.keras.layers.Dense(hparams.embedding_size, name="dense")
        self._act_fn = ACT2FN[hparams.hidden_act]

    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self._act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


@BaseLayer.register("electra")
class Electra(BaseLayer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(Electra, self).__init__(hparams, **kwargs)
        self.embeddings = ElectraEmbeddings(hparams.config, name="embeddings")

        if hparams.config.embedding_size != hparams.config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(hparams.config.hidden_size, name="embeddings_project")
        self.encoder = Transformer(hparams.config, name="encoder")

    def get_input_embeddings(self):
        return self.embeddings

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self._hparams.config.num_hidden_layers

        return head_mask

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = get_shape(input_ids)
        elif inputs_embeds is not None:
            input_shape = get_shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask)

        hidden_states = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states, training=training)

        encoder_outputs = self.encoder([hidden_states, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0]

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs
