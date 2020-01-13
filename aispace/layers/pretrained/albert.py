# -*- coding: utf-8 -*-
# @Time    : 2019-11-29 14:19
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : albert.py


import tensorflow as tf

from aispace.layers.attentions import MultiHeadAttention
from aispace.utils.hparams import Hparams
from aispace.utils.tf_utils import get_initializer
from aispace.layers.activations import ACT2FN
from aispace.layers.embeddings import AlbertEmbedding, AlbertEmbeddingV2
from aispace.layers.base_layer import BaseLayer


class AlbertAttention(MultiHeadAttention):
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertAttention, self).__init__(hparams, **kwargs)

        self.hidden_size = hparams.hidden_size
        self.initializer_range = hparams.initializer_range
        self.layer_norm_eps = hparams.layer_norm_eps

        self.dense = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=get_initializer(self.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='LayerNorm')

    def call(self, inputs, training=False):
        input_tensor, attention_mask, head_mask = inputs

        batch_size = tf.shape(input_tensor)[0]
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # scale attention_scores
        dk = tf.cast(tf.shape(key_layer)[-1], tf.float32)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
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

        self_outputs = (context_layer, attention_probs) if self.output_attentions else (
            context_layer,)

        hidden_states = self_outputs[0]

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        attention_output = self.LayerNorm(hidden_states + input_tensor)

        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class AlbertLayer(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertLayer, self).__init__(**kwargs)
        self.intermediate_size = hparams.intermediate_size
        self.initializer_range = hparams.initializer_range
        self.hidden_act = hparams.hidden_act
        self.hidden_size = hparams.hidden_size
        self.layer_norm_eps = hparams.layer_norm_eps
        self.hidden_dropout_prob = hparams.hidden_dropout_prob

        self.attention = AlbertAttention(hparams, name='attention')
        self.ffn = tf.keras.layers.Dense(
            self.intermediate_size,
            kernel_initializer=get_initializer(self.initializer_range),
            name='ffn'
        )
        self.activation = ACT2FN[self.hidden_act]
        self.ffn_output = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=get_initializer(self.initializer_range),
            name='ffn_output'
        )
        self.full_layer_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='full_layer_layer_norm'
        )
        self.dropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask, head_mask], training=training)
        ffn_output = self.ffn(attention_outputs[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.full_layer_layer_norm(
            ffn_output + attention_outputs[0])

        # add attentions if we output them
        outputs = (hidden_states,) + attention_outputs[1:]
        return outputs


class AlbertLayerGroup(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertLayerGroup, self).__init__(**kwargs)

        self.output_attentions = hparams.output_attentions
        self.output_hidden_states = hparams.output_hidden_states
        self.inner_group_num = hparams.inner_group_num

        self.albert_layers = [AlbertLayer(hparams, name="albert_layers_._{}".format(
            i)) for i in range(self.inner_group_num)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(
                [hidden_states, attention_mask, head_mask[layer_index]], training=training)
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        # last-layer hidden state, (layer hidden states), (layer attentions)
        return outputs


class AlbertTransformer(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertTransformer, self).__init__(**kwargs)

        self.output_attentions = hparams.output_attentions
        self.output_hidden_states = hparams.output_hidden_states
        self.hidden_size = hparams.hidden_size
        self.initializer_range = hparams.initializer_range
        self.num_hidden_groups = hparams.num_hidden_groups
        self.num_hidden_layers = hparams.num_hidden_layers

        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=get_initializer(self.initializer_range),
            name='embedding_hidden_mapping_in'
        )
        self.albert_layer_groups = [AlbertLayerGroup(
            hparams, name="albert_layer_groups_._{}".format(i)) for i in range(self.num_hidden_groups)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs

        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(
                self.num_hidden_layers / self.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(
                i / (self.num_hidden_layers / self.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                [hidden_states, attention_mask, head_mask[group_idx*layers_per_group:(group_idx+1)*layers_per_group]], training=training)
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class AlbertTransformerV2(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertTransformerV2, self).__init__(**kwargs)

        self.output_attentions = hparams.output_attentions
        self.output_hidden_states = hparams.output_hidden_states
        self.hidden_size = hparams.hidden_size
        self.initializer_range = hparams.initializer_range
        self.num_hidden_groups = hparams.num_hidden_groups
        self.num_hidden_layers = hparams.num_hidden_layers
        self.albert_layer_groups = [AlbertLayerGroup(
            hparams, name="albert_layer_groups_._{}".format(i)) for i in range(self.num_hidden_groups)]

    def call(self, inputs, training=False):
        hidden_states, attention_mask, head_mask = inputs
        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(
                self.num_hidden_layers / self.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(
                i / (self.num_hidden_layers / self.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                [hidden_states, attention_mask, head_mask[group_idx*layers_per_group:(group_idx+1)*layers_per_group]], training=training)
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class AlbertMLMHead(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, input_embeddings, **kwargs):
        super(AlbertMLMHead, self).__init__(**kwargs)
        self.vocab_size = hparams.vocab_size
        self.embedding_size = hparams.embedding_size
        self.initializer_range = hparams.initializer_range
        self.hidden_act = hparams.hidden_act
        self.layer_norm_eps = hparams.layer_norm_eps

        self.dense = tf.keras.layers.Dense(
            self.embedding_size,
            kernel_initializer=get_initializer(self.initializer_range),
            name='dense'
        )
        self.activation = ACT2FN[self.hidden_act]
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='LayerNorm'
        )
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        self.decoder_bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='decoder/bias')
        super(AlbertMLMHead, self).build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states, mode="linear") + self.decoder_bias
        hidden_states = hidden_states + self.bias
        return hidden_states


@BaseLayer.register("albert")
class Albert(BaseLayer):
    """google version

    """
    def __init__(self, hparams: Hparams, **kwargs):
        super(Albert, self).__init__(hparams, **kwargs)
        self.num_hidden_layers = hparams.config.num_hidden_layers
        self.hidden_size = hparams.config.hidden_size
        self.initializer_range = hparams.config.initializer_range

        self.embeddings = AlbertEmbedding(hparams.config, name="embeddings")
        self.encoder = AlbertTransformer(hparams.config, name="encoder")
        self.pooler = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=get_initializer(self.initializer_range),
            activation='tanh',
            name='pooler'
        )

    def get_input_embeddings(self):
        return self.embeddings

    def call(self, inputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
             inputs_embeds=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = tf.shape(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

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

        embedding_output = self.embeddings(
            [input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output[:, 0])

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


@BaseLayer.register("albert_brightmart")
class AlbertBrightmart(BaseLayer):
    """brightmart version

    ref: https://github.com/brightmart/albert_zh
    """
    def __init__(self, hparams: Hparams, **kwargs):
        super(AlbertBrightmart, self).__init__(hparams, **kwargs)
        self.num_hidden_layers = hparams.config.num_hidden_layers
        self.hidden_size = hparams.config.hidden_size
        self.initializer_range = hparams.config.initializer_range

        self.embeddings = AlbertEmbeddingV2(hparams.config, name="embeddings")
        self.encoder = AlbertTransformerV2(hparams.config, name="encoder")
        self.pooler = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=get_initializer(self.initializer_range),
            activation='tanh',
            name='pooler'
        )

    def get_input_embeddings(self):
        return self.embeddings

    def call(self, inputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
             inputs_embeds=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = tf.shape(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

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

        embedding_output = self.embeddings(
            [input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output[:, 0])

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs