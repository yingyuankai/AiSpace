# !/usr/bin/env python
# coding=utf-8
# @Time    : 2020/4/25 18:07
# @Author  : yingyuankai@aliyun.com
# @File    : bert_for_qa.py


import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers import BaseLayer
from aispace.layers.activations import ACT2FN
from aispace.utils.tf_utils import get_initializer

__all__ = [
    "BertForQA"
]


@BaseModel.register("bert_for_qa")
class BertForQA(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForQA, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        model_hparams = hparams.model_attributes
        self.start_n_top = model_hparams.start_n_top
        self.seq_len = hparams.dataset.tokenizer.max_len

        # assert pretrained_hparams.norm_name in ['bert', 'albert', 'albert_brightmart', "ernie"], \
        #     ValueError(f"{pretrained_hparams.norm_name} not be supported.")
        self.encode_pretrained = BaseLayer.by_name(pretrained_hparams.norm_name)(pretrained_hparams)

        self.qa_layer = BaseLayer.by_name(model_hparams.qa_layer_name)(
            model_hparams.hidden_size,
            self.seq_len,
            self.start_n_top,
            self.start_n_top,
            get_initializer(model_hparams.initializer_range),
            model_hparams.hidden_dropout_prob)

    def call(self, inputs, **kwargs):
        is_training = kwargs.get("training", False)
        new_inputs = {
            "input_ids": tf.cast(inputs['input_ids'], tf.int32),
            'token_type_ids': inputs['token_type_ids'],
            "attention_mask": inputs['attention_mask']
        }
        encode_repr = self.encode_pretrained(new_inputs, **kwargs)
        seq_output = encode_repr[0]  # [b, l, h]
        cls_output = encode_repr[1]  # [b, h]
        passage_mask = inputs['p_mask']

        if is_training:
            start_position = inputs['start_position']
        else:
            start_position = None

        outputs = self.qa_layer([seq_output, cls_output, passage_mask, start_position], training=is_training)

        return outputs + (inputs['unique_id'], )
