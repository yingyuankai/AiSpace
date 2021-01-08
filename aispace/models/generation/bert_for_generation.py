# -*- coding: utf-8 -*-
# @Time    : 1/7/21 10:14 AM
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_generation.py

import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.generation_tf_utils import TFGenerationMixin
from aispace.models.base_model import BaseModel
from aispace.layers.pretrained.bert import Bert
from aispace.utils.tf_utils import get_initializer
from aispace.layers import BaseLayer

__all__ = [
    'BertForTextGeneration'
]


@BaseModel.register("bert_for_text_generation")
class BertForTextGeneration(BaseModel, TFGenerationMixin):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForTextGeneration, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained

        assert pretrained_hparams.norm_name in ['gpt2'], \
            ValueError(f"{pretrained_hparams.norm_name} not be supported.")
        self.transformer = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)

    def call(self, inputs, **kwargs):
        transformer_outputs = self.transformer(inputs, **kwargs)
        hidden_states = transformer_outputs[0]
        logits = self.transformer.wte(hidden_states, mode="linear")

        return logits

    def deploy(self):
        pass
