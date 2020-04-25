# !/usr/bin/env python
# coding=utf-8
# @Time    : 2020/4/25 18:07
# @Author  : yingyuankai@aliyun.com
# @File    : bert_for_qa.py


import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers.pretrained.bert import Bert
from aispace.utils.tf_utils import get_initializer
from aispace.layers import BaseLayer


@BaseModel.register("bert_for_qa")
class BertForQA(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForQA, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes

        assert pretrained_hparams.norm_name in ['bert', 'albert', 'albert_brightmart', "ernie"], \
            ValueError(f"{pretrained_hparams.norm_name} not be supported.")
        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)
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
        self.qa_outputs = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(model_hparams.initializer_range), name="qa_outputs"
        )
        self.classifier = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="classifier"
        )

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        seq_output = outputs[0]
        cls_output = outputs[1]

        # pos predict
        project1 = self.project1(seq_output)
        project1 = self.dropout(project1, training=kwargs.get('training', False))
        qa_logits = self.qa_outputs(project1)
        start_logits, end_logits = tf.split(qa_logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # is_impossibale predict
        project2 = self.project2(cls_output)
        project2 = self.dropout(project2, training=kwargs.get('training', False))
        is_impossible_logits = self.classifier(project2)

        outputs = (start_logits, end_logits, is_impossible_logits) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


