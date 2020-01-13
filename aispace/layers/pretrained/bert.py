# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 19:34
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert.py

__all__ = [
    "Bert",
    "Ernie",
    "BertMLMTask",
    "BertNSPTask"
]

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.layers.encoders.transformer import Transformer
from aispace.layers.embeddings.bert_embedding import BertEmbedding
from aispace.layers.embeddings.ernie_embedding import ErnieEmbedding
from aispace.utils.tf_utils import get_initializer, get_shape
from aispace.layers.activations import ACT2FN
from aispace.layers.base_layer import BaseLayer


class BertPooler(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertPooler, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            hparams.hidden_size,
            kernel_initializer=get_initializer(hparams.initializer_range),
            activation='tanh',
            name='dense'
        )

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class BertPredicationTaskTransform(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertPredicationTaskTransform, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hparams.hidden_size,
                                           kernel_initializer=get_initializer(hparams.initializer_range),
                                           name='dense')
        self.transform_act_fn = ACT2FN[hparams.hidden_act]
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=hparams.layer_norm_eps,
            name="LayerNorm"
        )

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionTask(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, input_embeddings, **kwargs):
        super(BertLMPredictionTask, self).__init__(**kwargs)
        self.vocab_size = hparams.vocab_size
        self.transform = BertPredicationTaskTransform(hparams, name='transform')

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        super(BertLMPredictionTask, self).build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


class BertMLMTask(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, input_embeddings, **kwargs):
        super(BertMLMTask, self).__init__(**kwargs)
        self.predictions = \
            BertLMPredictionTask(hparams, input_embeddings, name='predictions')

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertNSPTask(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super(BertNSPTask, self).__init__(**kwargs)
        self.seq_relationship = tf.keras.layers.Dense(
            2,
            kernel_initializer=get_initializer(hparams.initializer_range),
            name='seq_relationship'
        )

    def call(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


@BaseLayer.register("bert")
class Bert(BaseLayer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(Bert, self).__init__(hparams, **kwargs)
        self.num_hidden_layers = hparams.config.num_hidden_layers
        self.embeddings = BertEmbedding(hparams.config, name="embeddings")
        self.encoder = Transformer(hparams.config, name="encoder")
        self.pooler = BertPooler(hparams.config, name="pooler")

    def call(self, inputs, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            assert len(inputs) <= 5, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            assert len(inputs) <= 5, "Too many inputs."
        else:
            input_ids = inputs

        if attention_mask is None:
            attention_mask = tf.fill(get_shape(input_ids), 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(get_shape(input_ids), 0)

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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if not head_mask is None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        embedding_output = self.embeddings([input_ids, position_ids, token_type_ids], training=training)
        encoder_outputs = self.encoder([embedding_output, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@BaseLayer.register("ernie")
class Ernie(BaseLayer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(Ernie, self).__init__(hparams, **kwargs)
        self.num_hidden_layers = hparams.config.num_hidden_layers
        self.embeddings = ErnieEmbedding(hparams.config, name="embeddings")
        self.encoder = Transformer(hparams.config, name="encoder")
        self.pooler = BertPooler(hparams.config, name="pooler")

    def call(self, inputs, token_type_ids=None, attention_mask=None, position_ids=None, task_type_ids=None, head_mask=None,
             training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            task_ids = inputs[4] if len(inputs) > 4 else task_type_ids
            head_mask = inputs[4] if len(inputs) > 5 else head_mask
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            task_type_ids = inputs.get("task_type_ids", task_type_ids)
            head_mask = inputs.get('head_mask', head_mask)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if attention_mask is None:
            attention_mask = tf.fill(get_shape(input_ids), 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(get_shape(input_ids), 0)

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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if not head_mask is None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        embedding_output = self.embeddings([input_ids, position_ids, token_type_ids, task_type_ids], training=training)
        encoder_outputs = self.encoder([embedding_output, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
