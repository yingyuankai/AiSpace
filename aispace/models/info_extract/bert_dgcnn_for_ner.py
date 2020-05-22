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
from aispace.layers.encoders import DgcnnBlock


__all__ = [
    "BertDgcnnForNer"
]


@BaseModel.register("bert_dgcnn_for_ner")
class BertDgcnnForNer(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertDgcnnForNer, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.num_labels = hparams.dataset.outputs[0].num
        self.pos_num = hparams.dataset.inputs[-1].num
        self.initializer_range = model_hparams.initializer_range

        self.pos_embeddings = tf.keras.layers.Embedding(
            self.pos_num,
            32,
            embeddings_initializer=get_initializer(model_hparams.initializer_range),
            name="pos_embedding"
        )

        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)
        self.dropout = tf.keras.layers.Dropout(
            model_hparams.hidden_dropout_prob
        )
        self.project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="project"
        )
        self.fusion_project = tf.keras.layers.Dense(
            model_hparams.hidden_size,
            kernel_initializer=get_initializer(model_hparams.initializer_range),
            name="fusion_project"
        )
        self.dgcnn_encoder = DgcnnBlock(model_hparams.hidden_size, [3, 3, 3], [1, 2, 4], name="trigger_dgcnn_encoder")
        self.ner_output = tf.keras.layers.Dense(self.num_labels,
                                                kernel_initializer=get_initializer(model_hparams.initializer_range),
                                                name='ner_output')
        self.crf = CRFLayer(self.num_labels, self.initializer_range, label_mask=hparams.label_mask, name="crf_output")

    def call(self, inputs, **kwargs):
        training = kwargs.get('training', False)
        # sequence length
        input_ids = inputs['input_ids']
        input_lengths = get_sequence_length(input_ids)
        # bert encode
        bert_encode = self.bert(inputs, **kwargs)
        seq_output = bert_encode[0]

        # pos encode
        pos_ids = inputs["pos"]
        pos_repr = self.pos_embeddings(pos_ids)
        pos_repr = self.dropout(pos_repr, training=training)

        # feature fusion
        feature_fusion = tf.concat([seq_output, pos_repr], -1)
        # feature_fusion = seq_output
        feature_fusion = self.fusion_project(feature_fusion)
        feature_fusion = self.dropout(feature_fusion, training=training)

        # dgcnn
        feature_fusion = self.dgcnn_encoder(feature_fusion, training=training)
        # project
        project = self.project(feature_fusion)
        project = self.dropout(project, training=training)
        logits = self.ner_output(project)
        # crf
        viterbi, _ = self.crf([logits, input_lengths])
        return tf.keras.backend.in_train_phase(logits, tf.one_hot(viterbi, self.num_labels), training=training)

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

