# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 20:44
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_sequence_classification.py


import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers.encoders import TextcnnBlock
from aispace.utils.tf_utils import get_initializer
from aispace.layers import BaseLayer


@BaseModel.register("textcnn_for_classification")
class TextcnnForSeqClassification(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(TextcnnForSeqClassification, self).__init__(hparams, **kwargs)
        self.num_lables = hparams.dataset.outputs[0].num
        model_hparams = hparams.model_attributes

        self.embeddings = tf.keras.layers.Embedding(
            model_hparams.vocab_size,
            model_hparams.hidden_size,
            embeddings_initializer=get_initializer(model_hparams.initializer_range),
            name="embeddings"
        )

        self.encoder = TextcnnBlock(model_hparams.filters, model_hparams.windows, model_hparams.initializer_range, name="textcnn_encoder")
        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        self.project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project"
        )
        self.classifier = tf.keras.layers.Dense(
            self.num_lables,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="classifier"
        )

    def call(self, inputs, **kwargs):
        emb = self.embeddings(inputs['input_ids'])
        output = self.encoder(emb, **kwargs)

        project = self.project(output)

        project = self.dropout(project, training=kwargs.get('training', False))

        logits = self.classifier(project)

        outputs = (logits,)

        return outputs  # logits, (hidden_states), (attentions)



