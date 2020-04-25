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
from aispace.utils.tf_utils import get_sequence_length
from aispace.layers import BaseLayer


__all__ = [
    "BertForTriggerNer"
]


@BaseModel.register("bert_for_trigger_ner")
class BertForTriggerNer(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForTriggerNer, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.num_ner_labels = hparams.dataset.outputs[0].num
        self.num_class_labels = hparams.dataset.outputs[1].num
        self.initializer_range = model_hparams.initializer_range

        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)
        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        self.project_for_ner = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project_for_ner"
        )
        self.ner_output = tf.keras.layers.Dense(self.num_ner_labels,
                                                kernel_initializer=get_initializer(model_hparams.initializer_range),
                                                name='ner_output')
        self.crf = CRFLayer(self.num_ner_labels, self.initializer_range, name="crf_output")

        self.project_for_class = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project_for_class"
        )
        self.class_output = tf.keras.layers.Dense(self.num_class_labels,
                                                kernel_initializer=get_initializer(model_hparams.initializer_range),
                                                name='class_output')

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)
        # sequence length
        input_ids = inputs['input_ids']
        input_lengths = get_sequence_length(input_ids)
        # bert encode
        bert_encode = self.bert(inputs, **kwargs)
        seq_output = bert_encode[0]
        cls_output = bert_encode[1]
        # project for ner
        project1 = self.project_for_ner(seq_output)
        project1 = self.dropout(project1, training=training)
        ner_logits = self.ner_output(project1)
        # crf
        viterbi, _ = self.crf([ner_logits, input_lengths])
        ner_output = tf.keras.backend.in_train_phase(ner_logits, tf.one_hot(viterbi, self.num_ner_labels), training=training)

        # project for classify
        project2 = self.project_for_class(cls_output)
        project2 = self.dropout(project2, training=training)
        class_logits = self.class_output(project2)
        return ner_output, class_logits

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

