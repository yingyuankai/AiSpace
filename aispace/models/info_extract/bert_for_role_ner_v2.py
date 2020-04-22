# -*- coding: utf-8 -*-
# @Time    : 2019-11-11 19:33
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_ner.py

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers.pretrained.bert import Bert
from aispace.layers.decoders import CRFLayer
from aispace.layers.encoders import Bilstm
from aispace.utils.tf_utils import get_initializer
from aispace.utils.tf_utils import get_sequence_length, get_shape, tf_gather
from aispace.layers import BaseLayer
from aispace.layers.attentions import MultiHeadAttention


__all__ = [
    "BertForRoleNerV2"
]


@BaseModel.register("bert_for_role_ner_v2")
class BertForRoleNerV2(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForRoleNerV2, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.num_labels = hparams.dataset.outputs[0].num
        self.initializer_range = model_hparams.initializer_range

        # self.pre_pos_embeddings = tf.keras.layers.Embedding(
        #     hparams.dataset.tokenizer.max_len,
        #     model_hparams.pos_emb_size,
        #     embeddings_initializer=get_initializer(model_hparams.initializer_range),
        #     name='pre_position_embeddings')
        # self.post_pos_embeddings = tf.keras.layers.Embedding(
        #     hparams.dataset.tokenizer.max_len,
        #     model_hparams.pos_emb_size,
        #     embeddings_initializer=get_initializer(model_hparams.initializer_range),
        #     name='post_position_embeddings')

        # self.bert = Bert(pretrained_hparams, name='bert')
        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)

        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        # self.bilstm = Bilstm(model_hparams.hidden_size, model_hparams.hidden_dropout_prob, name="bilstm")

        # self.attention = MultiHeadAttention(model_hparams, name="attention_fusion")
        self.type_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="type_project"
        )
        self.type_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="type_project"
        )
        self.passage_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="passage_project"
        )
        self.trigger_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="trigger_project"
        )
        self.output_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="output_project"
        )
        self.ner_output = tf.keras.layers.Dense(self.num_labels,
                                                kernel_initializer=get_initializer(model_hparams.initializer_range),
                                                name='ner_output')
        self.crf = CRFLayer(self.num_labels, self.initializer_range, name="crf_output")

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)
        # sequence length
        input_ids = inputs['input_ids']
        input_lengths = get_sequence_length(input_ids)
        shape = get_shape(input_ids)

        # type bert encode
        type_input = {
            "input_ids": inputs["type_input_ids"],
            "token_type_ids": inputs["type_token_type_ids"],
            "attention_mask": inputs["type_attention_mask"]
        }
        type_bert_encode = self.bert(type_input, **kwargs)
        type_repr = type_bert_encode[1]
        type_repr = self.type_project(type_repr)
        type_repr = self.dropout(type_repr, training=training)
        type_repr = tf.tile(tf.expand_dims(type_repr, 1), [1, shape[1], 1])

        # passage bert encode
        passage_input = {
            "input_ids": inputs["input_ids"],
            "token_type_ids": inputs["token_type_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        passage_bert_encode = self.bert(passage_input, **kwargs)
        seq_output = passage_bert_encode[0]
        seq_output = self.passage_project(seq_output)
        seq_output = self.dropout(seq_output, training=training)

        # feature fusion (word, event type, relative pos)
        # pre relative pos repr
        # pre_pos_repr = self.pre_pos_embeddings(inputs["pre_rel_pos"])
        # post relative pos repr
        # pos_pos_repr = self.post_pos_embeddings(inputs["pos_rel_pos"])

        # fusion with event type
        seq_output = seq_output + type_repr

        # trigger repr
        trigger_span = tf.reshape(inputs["trigger_span"], [shape[0], 2])
        trigger_repr = tf_gather(seq_output, trigger_span)  # [batch_size, 2, hidden_size]
        trigger_repr = tf.unstack(trigger_repr, axis=1)
        trigger_repr = tf.concat(trigger_repr, axis=-1)
        trigger_repr = self.trigger_project(trigger_repr)
        trigger_repr = self.dropout(trigger_repr, training=training)
        trigger_repr = tf.tile(tf.expand_dims(trigger_repr, 1), [1, shape[1], 1])
        seq_output = seq_output + trigger_repr

        # project for output1
        project = self.output_project(seq_output)
        project = self.dropout(project, training=training)
        logits = self.ner_output(project)

        # ner logits add mask
        label_mask = inputs["label_mask"]
        label_mask = tf.tile(tf.expand_dims(label_mask, 1), [1, shape[1], 1])
        label_mask = tf.cast(label_mask, tf.float32)
        label_mask = (1.0 - label_mask) * -10000.0
        logits = logits + label_mask

        # crf
        viterbi, _ = self.crf([logits, input_lengths])
        output1 = tf.keras.backend.in_train_phase(logits, tf.one_hot(viterbi, self.num_labels), training=training)

        return output1

    def crf_loss(self, config):
        return self.crf.loss

    def deploy(self):
        from aispace.datasets.tokenizer import BertTokenizer
        from .bento_services import RoleBertNerService
        tokenizer = BertTokenizer(self._hparams.dataset.tokenizer)
        bento_service = \
            RoleBertNerService.pack(
                model=self,
                tokenizer=tokenizer,
                hparams=self._hparams,
            )
        saved_path = bento_service.save(self._hparams.get_deploy_dir())
        return saved_path

