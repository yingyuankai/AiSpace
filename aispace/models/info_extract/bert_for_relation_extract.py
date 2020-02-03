# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 16:38
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_relation_extract.py

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers.pretrained.bert import Bert
from aispace.layers.attentions import MultiHeadAttention
from aispace.utils.tf_utils import get_initializer, get_shape, tf_gather


@BaseModel.register('bert_for_relation_extract')
class BertForRelationExtract(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForRelationExtract, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.hidden_size = model_hparams.hidden_size
        self.num_labels = hparams.dataset.outputs[0].num
        self.initializer_range = model_hparams.initializer_range

        self.bert = Bert(pretrained_hparams, name='bert')
        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        self.project1 = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project1"
        )
        self.project2 = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project2"
        )
        self.project3 = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project3"
        )
        self.project4 = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project4"
        )
        self.project5 = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project5"
        )
        self.e1_attention = MultiHeadAttention(model_hparams, name="entity1_attention_fusion")
        self.e2_attention = MultiHeadAttention(model_hparams, name="entity2_attention_fusion")
        self.attention = MultiHeadAttention(model_hparams, name="attention_fusion")
        self.classifer = tf.keras.layers.Dense(
            self.num_labels,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="classifier"
        )

    def call(self, inputs, **kwargs):
        # inputs
        # [batch_size, 2]
        entity_span_start = inputs['entity_span_start']
        entity_span_end = inputs['entity_span_end']
        # entity_labels = inputs['entity_labels']
        batch_size, max_entity_num = get_shape(entity_span_start)
        entity_span_start = tf.reshape(entity_span_start, [batch_size, 2])
        entity_span_end = tf.reshape(entity_span_end, [batch_size, 2])

        # bert encode [batch_size, seq_len, hidden_size]
        bert_encode = self.bert(
            inputs=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
            attention_mask=inputs['attention_mask'], **kwargs)
        seq_output = bert_encode[0]

        # build repr for entities
        entity_start_repr = tf_gather(seq_output, entity_span_start)      # [batch_size, 2, hidden_size]
        entity_end_repr = tf_gather(seq_output, entity_span_end)          # [batch_size, 2, hidden_size]
        # [batch_size, 2, hidden_size * 2]
        entity_repr = tf.concat([entity_start_repr, entity_end_repr], -1)
        # [batch_size, 2, hidden_size]
        entity_repr = self.project1(entity_repr)
        entity_repr = self.dropout(entity_repr, training=kwargs.get('training', False))

        # fusion of every entity pairs
        entity_reprs = tf.unstack(entity_repr, axis=1)
        entity_repr1_r = tf.reshape(entity_reprs[0], [batch_size, self.hidden_size])
        entity_repr2_r = tf.reshape(entity_reprs[1], [batch_size, self.hidden_size])
        # 1 just entity_pair repr
        # entity_pair_repr = tf.concat([entity_repr1, entity_repr2], -1)
        # entity_pair_repr = self.project2(entity_pair_repr)
        # entity_pair_repr = self.dropout(entity_pair_repr, training=kwargs.get('training', False))

        # 2 combine seq and entity_pair reprs
        # cls_output = bert_encode[1]
        # entity_pair_repr = tf.concat([cls_output, entity_repr1, entity_repr2], -1)
        # entity_pair_repr = self.project2(entity_pair_repr)
        # entity_pair_repr = self.dropout(entity_pair_repr, training=kwargs.get('training', False))

        # 3 attention
        # attention rep for entities
        entity_repr1 = self.e1_attention(tf.expand_dims(entity_repr1_r, 1), seq_output, seq_output, training=kwargs.get('training', False))
        entity_repr1 = tf.reshape(entity_repr1, [batch_size, self.hidden_size])
        entity_repr1 = entity_repr1 + entity_repr1_r
        entity_repr2 = self.e2_attention(tf.expand_dims(entity_repr2_r, 1), seq_output, seq_output, training=kwargs.get('training', False))
        entity_repr2 = tf.reshape(entity_repr2, [batch_size, self.hidden_size])
        entity_repr2 = entity_repr2 + entity_repr2_r

        # relation rep
        entity_pair_repr_0 = tf.concat([entity_repr1, entity_repr2, entity_repr1 - entity_repr2], -1)
        entity_pair_repr = self.project2(entity_pair_repr_0)
        entity_pair_repr1 = self.dropout(entity_pair_repr, training=kwargs.get('training', False))
        entity_pair_repr = tf.expand_dims(entity_pair_repr1, 1)

        # attention rep for relation rep
        entity_pair_repr = self.attention(entity_pair_repr, seq_output, seq_output, training=kwargs.get('training', False))
        entity_pair_repr = tf.reshape(entity_pair_repr[0], [batch_size, self.hidden_size])
        entity_pair_repr = entity_pair_repr1 + entity_pair_repr
        entity_pair_repr = tf.concat([entity_pair_repr, bert_encode[1]], -1)
        entity_pair_repr = self.project3(entity_pair_repr)
        entity_pair_repr = self.project4(entity_pair_repr)
        entity_pair_repr = self.project5(entity_pair_repr)
        entity_pair_repr = self.dropout(entity_pair_repr, training=kwargs.get("training", False))

        # classifer
        logits = self.classifer(entity_pair_repr)
        return logits

    def deploy(self):
        from aispace.datasets.tokenizer import BertTokenizer
        from .bento_services import BertRelationClassificationService
        tokenizer = BertTokenizer(self._hparams.dataset.tokenizer)
        bento_service = \
            BertRelationClassificationService.pack(
                model=self,
                tokenizer=tokenizer,
                hparams=self._hparams,
            )
        saved_path = bento_service.save(self._hparams.get_deploy_dir())
        return saved_path
