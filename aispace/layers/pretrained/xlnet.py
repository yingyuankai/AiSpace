# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 19:35
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : xlnet.py

__all__ = [
    'XLNet'
]

import tensorflow as tf

from aispace.utils.tf_utils import get_initializer, get_shape
from aispace.utils.hparams import Hparams
from aispace.layers.activations import ACT2FN
from aispace.layers.base_layer import BaseLayer
from aispace.layers.embeddings import SharedEmbeddings
from aispace.layers.decoders import SequenceSummary
from aispace.utils.tf_utils import get_shape


class XLNetRelativeAttention(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(XLNetRelativeAttention, self).__init__(**kwargs)
        self.output_attentions = hparams.output_attentions

        if hparams.d_model % hparams.n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hparams.d_model, hparams.n_head))

        self.n_head = hparams.n_head
        self.d_head = hparams.d_head
        self.d_model = hparams.d_model
        self.scale = 1 / (hparams.d_head ** 0.5)
        self.initializer_range = hparams.initializer_range

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=hparams.layer_norm_eps, name='layer_norm')
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)

    def build(self, input_shape):
        initializer = get_initializer(self.initializer_range)
        self.q = self.add_weight(shape=(self.d_model, self.n_head, self.d_head),
                                 initializer=initializer,
                                 trainable=True, name='q')
        self.k = self.add_weight(shape=(self.d_model, self.n_head, self.d_head),
                                 initializer=initializer,
                                 trainable=True, name='k')
        self.v = self.add_weight(shape=(self.d_model, self.n_head, self.d_head),
                                 initializer=initializer,
                                 trainable=True, name='v')
        self.o = self.add_weight(shape=(self.d_model, self.n_head, self.d_head),
                                 initializer=initializer,
                                 trainable=True, name='o')
        self.r = self.add_weight(shape=(self.d_model, self.n_head, self.d_head),
                                 initializer=initializer,
                                 trainable=True, name='r')
        self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head),
                                        initializer='zeros',
                                        trainable=True, name='r_r_bias')
        self.r_s_bias = self.add_weight(shape=(self.n_head, self.d_head),
                                        initializer='zeros',
                                        trainable=True, name='r_s_bias')
        self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head),
                                        initializer='zeros',
                                        trainable=True, name='r_w_bias')
        self.seg_embed = self.add_weight(shape=(2, self.n_head, self.d_head),
                                        initializer=initializer,
                                        trainable=True, name='seg_embed')
        super(XLNetRelativeAttention, self).build(input_shape)

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = get_shape(x)

        x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        x = x[1:, ...]
        x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        x = x[:, 0:klen, :, :]
        # x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    def rel_attn_core(self, inputs, training=False):
        """Core relative positional attention operations."""

        q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask = inputs

        # content based attention score
        ac = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=get_shape(ac)[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            if attn_mask.dtype == tf.float16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = tf.nn.softmax(attn_score, axis=1)

        attn_prob = self.dropout(attn_prob, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # attention output
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        if self.output_attentions:
            return attn_vec, attn_prob

        return attn_vec

    def post_attention(self, inputs, residual=True, training=False):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        h, attn_vec = inputs

        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)

        attn_out = self.dropout(attn_out, training=training)

        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def call(self, inputs, training=False):
        (h, g, attn_mask_h, attn_mask_g,
         r, seg_mat, mems, target_mapping, head_mask) = inputs

        if g is not None:
            ###### Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.shape.ndims > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h

            # content-based key head
            k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.k)

            # content-based value head
            v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.v)

            # position-based key head
            k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)

            ##### h-stream
            # content-stream query head
            q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                [q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, head_mask],
                training=training)

            if self.output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention([h, attn_vec_h], training=training)

            ##### g-stream
            # query-stream query head
            q_head_g = tf.einsum('ibh,hnd->ibnd', g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    [q_head_g, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_g, head_mask],
                    training=training)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    [q_head_g, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_g, head_mask],
                    training=training)

                if self.output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention([g, attn_vec_g], training=training)

            if self.output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            ###### Multi-head attention with relative positional encoding
            if mems is not None and mems.shape.ndims > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h

            # content heads
            q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.v)

            # positional heads
            k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                [q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, head_mask],
                training=training)

            if self.output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention([h, attn_vec], training=training)
            output_g = None

        outputs = (output_h, output_g)
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(XLNetFeedForward, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=hparams.layer_norm_eps, name='layer_norm')
        self.layer_1 = tf.keras.layers.Dense(hparams.d_inner,
                                             kernel_initializer=get_initializer(hparams.initializer_range),
                                             name='layer_1')
        self.layer_2 = tf.keras.layers.Dense(hparams.d_model,
                                             kernel_initializer=get_initializer(hparams.initializer_range),
                                             name='layer_2')
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)

        self.activation_function = ACT2FN[hparams.ff_activation]

    def call(self, inp, training=False):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output, training=training)
        output = self.layer_2(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(XLNetLayer, self).__init__(**kwargs)
        self.rel_attn = XLNetRelativeAttention(hparams, name='rel_attn')
        self.ff = XLNetFeedForward(hparams, name='ff')
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)

    def call(self, inputs, training=False):
        outputs = self.rel_attn(inputs, training=training)
        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g, training=training)
        output_h = self.ff(output_h, training=training)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        return outputs


class XLNetLMHead(tf.keras.layers.Layer):
    def __init__(self, hparams: Hparams, input_embeddings, **kwargs):
        super(XLNetLMHead, self).__init__(**kwargs)
        self.vocab_size = hparams.vocab_size
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        super(XLNetLMHead, self).build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@BaseLayer.register("xlnet")
class XLNet(BaseLayer):
    def __init__(self, hparams: Hparams, **kwargs):
        super(XLNet, self).__init__(hparams, **kwargs)
        config = hparams.config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.output_past = config.output_past

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.use_bfloat16 = config.use_bfloat16
        self.initializer_range = config.initializer_range

        self.word_embedding = SharedEmbeddings(
            config.n_token,
            config.d_model,
            initializer_range=config.initializer_range,
            name='word_embedding'
        )
        self.layer = [XLNetLayer(config, name='layer_._{}'.format(i)) for i in range(config.n_layer)]
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.pooler = SequenceSummary(config, name="pooler")

    def get_input_embeddings(self):
        return self.word_embedding

    def build(self, input_shape):
        initializer = get_initializer(self.initializer_range)
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.d_model),
            initializer=initializer,
            trainable=True,
            name='mask_emb'
        )

    def create_mask(self, qlen, mlen, dtype=tf.float32):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        attn_mask = tf.ones([qlen, qlen], dtype=dtype)
        mask_u = tf.matrix_band_part(attn_mask, 0, -1)
        mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if self.same_length:
            mask_l = tf.matrix_band_part(attn_mask, -1, 0)
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        """cache hidden states into memory."""
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[:self.reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-self.mem_len:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[-self.mem_len:]

        return tf.stop_gradient(new_mem)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None, dtype=None):
        """create relative positional encoding."""
        freq_seq = tf.range(0, self.d_model, 2.0)
        if dtype is not None and dtype != tf.float32:
            freq_seq = tf.cast(freq_seq, dtype=dtype)
        inv_freq = 1 / (10000 ** (freq_seq / self.d_model))

        if self.attn_type == 'bi':
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == 'uni':
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            bwd_pos_seq = tf.range(-beg, -end, 1.0)

            if dtype is not None and dtype != tf.float32:
                fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
                bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)

            if bsz is not None:
                # With bi_data, the batch size should be divisible by 2.
                assert bsz%2 == 0
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            if dtype is not None and dtype != tf.float32:
                fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def call(self, inputs, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
            token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            mems = inputs[2] if len(inputs) > 2 else mems
            perm_mask = inputs[3] if len(inputs) > 3 else perm_mask
            target_mapping = inputs[4] if len(inputs) > 4 else target_mapping
            token_type_ids = inputs[5] if len(inputs) > 5 else token_type_ids
            input_mask = inputs[6] if len(inputs) > 6 else input_mask
            head_mask = inputs[7] if len(inputs) > 7 else head_mask
            inputs_embeds = inputs[8] if len(inputs) > 8 else inputs_embeds
            assert len(inputs) <= 9, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            mems = inputs.get('mems', mems)
            perm_mask = inputs.get('perm_mask', perm_mask)
            target_mapping = inputs.get('target_mapping', target_mapping)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            input_mask = inputs.get('input_mask', input_mask)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            assert len(inputs) <= 9, "Too many inputs."
        else:
            input_ids = inputs

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = tf.transpose(input_ids, perm=(1, 0))
            qlen, bsz = get_shape(input_ids)[:2]
        elif inputs_embeds is not None:
            inputs_embeds = tf.transpose(inputs_embeds, perm=(1, 0, 2))
            qlen, bsz = get_shape(inputs_embeds)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = tf.transpose(token_type_ids, perm=(1, 0)) if token_type_ids is not None else None
        input_mask = tf.transpose(input_mask, perm=(1, 0)) if input_mask is not None else None
        attention_mask = tf.transpose(attention_mask, perm=(1, 0)) if attention_mask is not None else None
        perm_mask = tf.transpose(perm_mask, perm=(1, 2, 0)) if perm_mask is not None else None
        target_mapping = tf.transpose(target_mapping, perm=(1, 2, 0)) if target_mapping is not None else None

        mlen = get_shape(mems[0])[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        dtype_float = tf.bfloat16 if self.use_bfloat16 else tf.float32

        ##### Attention mask
        # causal attention mask
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(self.attn_type))

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) " \
            "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)
            input_mask = 1. - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, bsz],
                                dtype=dtype_float)
            data_mask = tf.concat([mems_mask, data_mask], axis=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -tf.eye(qlen, dtype=dtype_float)
            non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=dtype_float), non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=dtype_float)
        else:
            non_tgt_mask = None

        ##### Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k, training=training)
        if target_mapping is not None:
            word_emb_q = tf.tile(self.mask_emb, [tf.shape(target_mapping)[0], bsz, 1])
        # else:  # We removed the inp_q input which was same as target mapping
        #     inp_q_ext = inp_q[:, :, None]
        #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q, training=training)
        else:
            output_g = None

        ##### Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
            cat_ids = tf.concat([mem_pad, token_type_ids], 0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = tf.cast(
                tf.logical_not(tf.equal(token_type_ids[:, None], cat_ids[None, :])),
                tf.int32)
            seg_mat = tf.one_hot(seg_mat, 2, dtype=dtype_float)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz, dtype=dtype_float)
        pos_emb = self.dropout(pos_emb, training=training)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []
        for i, layer_module in enumerate(self.layer):
            # cache new mems
            if self.mem_len is not None and self.mem_len > 0 and self.output_past:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module([output_h, output_g, non_tgt_mask, attn_mask,
                                    pos_emb, seg_mat, mems[i], target_mapping,
                                    head_mask[i]], training=training)
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h, training=training)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = tf.transpose(output, perm=(1, 0, 2))
        first_pooling = self.pooler(output)
        outputs = (output, first_pooling)

        if self.mem_len is not None and self.mem_len > 0 and self.output_past:
            outputs = outputs + (new_mems,)

        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(tf.transpose(h, perm=(1, 0, 2)) for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(tf.transpose(hs, perm=(1, 0, 2)) for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            attentions = tuple(tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, (new_mems), (hidden_states), (attentions)
