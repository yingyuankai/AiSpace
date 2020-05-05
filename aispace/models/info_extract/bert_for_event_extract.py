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
from aispace.utils.tf_utils import get_initializer, get_shape
from aispace.utils.tf_utils import get_sequence_length
from aispace.layers import BaseLayer
from aispace.layers.attentions import MultiHeadAttention

__all__ = [
    "BertForEventExtract"
]


@BaseModel.register("bert_for_event_extract")
class BertForEventExtract(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForEventExtract, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.trigger_num_labels = hparams.dataset.outputs[0].num
        self.role_num_labels = hparams.dataset.outputs[1].num
        self.rel_num_labels = hparams.dataset.tokenizer.max_len
        self.initializer_range = model_hparams.initializer_range
        self.hidden_size = model_hparams.hidden_size
        self.batch_size = hparams.training.batch_size

        self.relative_position_embeddings = tf.keras.layers.Embedding(
            self.rel_num_labels,
            64,
            embeddings_initializer=get_initializer(model_hparams.initializer_range),
            name="relative_position_embeddings"
        )

        # self.bert = Bert(pretrained_hparams, name='bert')
        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)
        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        # self.bilstm = Bilstm(model_hparams.hidden_size, model_hparams.hidden_dropout_prob, name="bilstm")
        self.trigger_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            activation=tf.keras.activations.relu,
            name="trigger_project"
        )
        self.role_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            activation=tf.keras.activations.relu,
            name="role_project"
        )
        self.rel_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            activation=tf.keras.activations.relu,
            name="rel_project"
        )
        self.trigger_ner_output = tf.keras.layers.Dense(self.trigger_num_labels,
                                                        kernel_initializer=get_initializer(
                                                            model_hparams.initializer_range),
                                                        activation=tf.keras.activations.relu,
                                                        name='trigger_ner_output')
        self.role_ner_output = tf.keras.layers.Dense(self.role_num_labels,
                                                     kernel_initializer=get_initializer(
                                                         model_hparams.initializer_range),
                                                     activation=tf.keras.activations.relu,
                                                     name='role_ner_output')
        self.rel_ner_output = tf.keras.layers.Dense(1,
                                                    kernel_initializer=get_initializer(
                                                        model_hparams.initializer_range),
                                                    activation=tf.keras.activations.relu,
                                                    name='rel_ner_output')
        self.trigger_attention = MultiHeadAttention(model_hparams, name="trigger_attention")
        self.role_attention = MultiHeadAttention(model_hparams, name="role_attention")

    @tf.function
    def get_relative_position(self, batch_size):
        seq_len = self.rel_num_labels
        positions = [[abs(j) for j in range(-i, seq_len - i)] for i in range(seq_len)]  # list [SEQ_LEN, SEQ_LEN]
        positions = tf.convert_to_tensor(positions, dtype=tf.int32)  # [seq_len, seq_len]
        positions = tf.tile(tf.expand_dims(positions, 0), [batch_size, 1, 1])  # [batch_size, seq_len, seq_len]
        position_repr = self.relative_position_embeddings(positions)  # [batch_size, seq_len, seq_len, hidden_size]
        return position_repr

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)
        # sequence length
        input_ids = inputs['input_ids']
        input_shape = get_shape(input_ids)
        input_lengths = get_sequence_length(input_ids)
        # bert encode
        bert_encode = self.bert(inputs, **kwargs)
        seq_output = bert_encode[0]
        # bilstm
        # seq_output = self.bilstm(seq_output)
        # project for trigger
        trigger_project = self.trigger_project(seq_output)
        trigger_project = self.dropout(trigger_project, training=training)
        # project for role
        role_project = self.role_project(seq_output)
        role_project = self.dropout(role_project, training=training)

        # trigger interactive
        trigger_fusion = self.trigger_attention(trigger_project, role_project, role_project, training=training)
        trigger_fusion = tf.reshape(trigger_fusion, [input_shape[0], input_shape[1], self.hidden_size])
        trigger_fusion = self.trigger_attention(trigger_fusion, trigger_fusion, trigger_fusion, training=training)
        trigger_fusion = tf.reshape(trigger_fusion, [input_shape[0], input_shape[1], self.hidden_size])
        trigger_fusion += trigger_project

        role_fusion = self.role_attention(role_project, trigger_project, trigger_project, training=training)
        role_fusion = tf.reshape(role_fusion, [input_shape[0], input_shape[1], self.hidden_size])
        role_fusion = self.role_attention(role_fusion, role_fusion, role_fusion, training=training)
        role_fusion = tf.reshape(role_fusion, [input_shape[0], input_shape[1], self.hidden_size])
        role_fusion += role_project

        # logits of trigger and role
        trigger_logits = self.trigger_ner_output(trigger_fusion)
        role_logits = self.role_ner_output(role_fusion)

        # trigger_mask
        trigger_mask = tf.argmax(trigger_logits, axis=-1)
        trigger_mask = tf.cast(trigger_mask > 0, tf.float32)
        trigger_mask = tf.tile(tf.expand_dims(trigger_mask, 1), [1, self.rel_num_labels, 1])
        role_mask = tf.argmax(role_logits, axis=-1)
        role_mask = tf.cast(role_mask > 0, tf.float32)
        role_mask = tf.tile(tf.expand_dims(role_mask, 2), [1, 1, self.rel_num_labels])
        trigger_role_mask = trigger_mask * role_mask

        # build mask [seq_len， seq_len]
        attention_mask = inputs['attention_mask']
        attention_mask = tf.cast(attention_mask, tf.float32)
        attention_mask1 = tf.tile(tf.expand_dims(attention_mask, 1), [1, self.rel_num_labels, 1])
        attention_mask2 = tf.tile(tf.expand_dims(attention_mask, 2), [1, 1, self.rel_num_labels])
        attention_mask = attention_mask1 * attention_mask2 * trigger_role_mask
        attention_mask = tf.cast(attention_mask, tf.float32)
        attention_mask = (1.0 - attention_mask) * -10000.0

        # relative positions
        relative_position_repr = self.get_relative_position(input_shape[0])

        # project for relation
        rel_reps1 = tf.tile(tf.expand_dims(trigger_fusion, 2), [1, 1, self.rel_num_labels, 1])
        rel_reps2 = tf.tile(tf.expand_dims(role_fusion, 2), [1, 1, self.rel_num_labels, 1])
        rel_reps = tf.concat([rel_reps1, rel_reps2, relative_position_repr], axis=-1)
        rel_project = self.rel_project(rel_reps)
        rel_project = self.dropout(rel_project, training=training)
        rel_logits = self.rel_ner_output(rel_project)
        rel_logits = tf.reshape(rel_logits, [input_shape[0], self.rel_num_labels, self.rel_num_labels])
        rel_logits += attention_mask
        rel_logits = tf.reshape(rel_logits, [input_shape[0], -1])

        return trigger_logits, role_logits, rel_logits

    def crf_loss(self, config):
        return self.crf.loss

    def deploy(self):
        from aispace.datasets.tokenizer import BertTokenizer
        from .bento_services import BertNerService
        tokenizer = BertTokenizer(self._hparams.dataset.tokenizer)
        bento_service = \
            BertNerService.pack(
                model=self,
                tokenizer=tokenizer,
                hparams=self._hparams,
            )
        saved_path = bento_service.save(self._hparams.get_deploy_dir())
        return saved_path
