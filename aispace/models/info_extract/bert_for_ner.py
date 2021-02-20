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
    "BertForNer"
]


@BaseModel.register("bert_for_ner")
class BertForNer(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForNer, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.num_labels = hparams.dataset.outputs[0].num
        self.initializer_range = model_hparams.initializer_range

        # self.bert = Bert(pretrained_hparams, name='bert')
        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)
        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        # self.bilstm = Bilstm(model_hparams.hidden_size, model_hparams.hidden_dropout_prob, name="bilstm")
        self.project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project"
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
        # bert encode
        bert_encode = self.bert(inputs, **kwargs)
        seq_output = bert_encode[0]
        # bilstm
        # seq_output = self.bilstm(seq_output)
        # project
        project = self.project(seq_output)
        project = self.dropout(project, training=training)
        logits = self.ner_output(project)
        # outputs = (logits,) + bert_encode[2:]
        # return outputs
        # crf
        viterbi, _ = self.crf([logits, input_lengths])
        return tf.keras.backend.in_train_phase(logits, tf.one_hot(viterbi, self.num_labels), training=training)

    def crf_loss(self, config):
        return self.crf.loss

    # def deploy(self):
    #     from aispace.datasets.tokenizer import BertTokenizer
    #     from .bento_services import BertNerWithTitleStatusService as BertNerService
    #     tokenizer = BertTokenizer(self._hparams.dataset.tokenizer)
    #     bento_service = BertNerService()
    #     bento_service.pack("model", self)
    #     bento_service.pack("tokenizer", tokenizer)
    #     bento_service.pack("hparams", self._hparams)
    #     saved_path = bento_service.save(self._hparams.get_deploy_dir())
    #     return saved_path

    def deploy(self):
        from aispace.datasets.tokenizer import BaseTokenizer
        from .bento_services import BertNerWithTitleStatusService as BertNerService
        tokenizer = BaseTokenizer.by_name(self._hparams.dataset.tokenizer.name)(self._hparams.dataset.tokenizer)
        bento_service = BertNerService()
        bento_service.pack("model", self)
        bento_service.pack("tokenizer", tokenizer)
        bento_service.pack("hparams", self._hparams)
        saved_path = bento_service.save(self._hparams.get_deploy_dir())
        return saved_path
