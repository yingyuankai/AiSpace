# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 20:44
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_sequence_classification.py


import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers.pretrained.bert import Bert
from aispace.utils.tf_utils import get_initializer
from aispace.layers import BaseLayer


@BaseModel.register("bert_for_classification")
class BertForSeqClassification(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForSeqClassification, self).__init__(hparams, **kwargs)
        self.num_lables = hparams.dataset.outputs[0].num
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes

        # self.bert = Bert(pretrained_hparams, name='bert')
        assert pretrained_hparams.norm_name in ['bert', 'albert', 'albert_brightmart', "ernie", "xlnet"], \
            ValueError(f"{pretrained_hparams.norm_name} not be supported.")
        self.bert = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)
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
        outputs = self.bert(inputs, **kwargs)

        pooled_output = outputs[1]
        project = self.project(pooled_output)

        project = self.dropout(project, training=kwargs.get('training', False))

        logits = self.classifier(project)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)

    def deploy(self):
        from aispace.datasets.tokenizer import BertTokenizer
        from .bento_services import BertTextClassificationService
        # tf.saved_model.save(self, self._hparams.get_saved_model_dir())
        tokenizer = BertTokenizer(self._hparams.dataset.tokenizer)
        bento_service = \
            BertTextClassificationService.pack(
                model=self,
                tokenizer=tokenizer,
                hparams=self._hparams,
            )
        saved_path = bento_service.save(self._hparams.get_deploy_dir())
        return saved_path


