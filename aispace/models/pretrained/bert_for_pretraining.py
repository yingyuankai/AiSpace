# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 20:24
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_pretraining.py


from aispace.utils.hparams import Hparams
from aispace.models.base_model import BaseModel
from aispace.layers.pretrained.bert import Bert, BertMLMTask, BertNSPTask


class BertForPreTraining(BaseModel):
    def __init__(self, hparams: Hparams, **kwargs):
        super(BertForPreTraining, self).__init__(hparams, **kwargs)
        pretrained_hparams = hparams.pretrained
        self.bert = Bert(pretrained_hparams, name="bert")
        self.nsp = BertNSPTask(pretrained_hparams.config, name="nsp___cls")
        self.mlm = BertMLMTask(pretrained_hparams.config, input_embeddings=self.bert.embeddings, name="mlm___cls")

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.mlm(sequence_output, training=kwargs.get('training', False))
        seq_relationship_score = self.nsp(pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
                                                                 2:]  # add hidden states and attention if they are here

        return outputs  # prediction_scores, seq_relationship_score, (hidden_states), (attentions)
