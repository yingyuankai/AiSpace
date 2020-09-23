# -*- coding: utf-8 -*-
# @Time    : 2019-12-03 20:41
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_sequence_classification.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    "BertTextClassificationService"
]

import os, sys
import logging
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" * 4)))

from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TensorflowSavedModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

import numpy as np
from scipy.special import softmax, expit

from aispace.datasets.tokenizer import BertTokenizer
from aispace.utils.hparams import Hparams


logger = logging.getLogger(__name__)

@artifacts([
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('tokenizer'),
        PickleArtifact("hparams"),
    ])
@env(auto_pip_dependencies=True)
class BertTextClassificationService(BentoService):

    def preprocessing(self, one_json):
        texts = one_json.get("text", '')
        if isinstance(texts, (list, tuple)):
            if len(texts) >= 2:
                encode = self.artifacts.tokenizer.encode(texts[0], texts[1])
            elif len(texts) == 1:
                encode = self.artifacts.tokenizer.encode(texts[0])
            else:
                return None, None, None
        elif isinstance(texts, str):
            encode = self.artifacts.tokenizer.encode(texts)
        else:
            raise NotImplementedError
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['segment_ids'], encode['input_mask']
        return input_ids, token_type_ids, attention_mask

    def decode_label_idx(self, idx):
        return self.artifacts.hparams.dataset.outputs[0].labels[idx]

    @api(JsonHandler, mb_max_latency=3000, mb_max_batch_size=20, batch=True)
    def label_predict(self, parsed_json):
        # print(parsed_json)
        # logger.info(parsed_json)
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": []
        }
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(itm for itm in list(map(self.preprocessing, parsed_json)) if itm[0] is not None)))
            # print(pre_input_data[0])
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
        else:
            raise NotImplementedError
        # else:  # expecting type(parsed_json) == dict:
        #     pre_input_data = self.preprocessing(parsed_json)
        #     if pre_input_data[0] is None:
        #         return []
        #     input_data['input_ids'].append(pre_input_data[0])
        #     input_data['token_type_ids'].append(pre_input_data[1])
        #     input_data['attention_mask'].append(pre_input_data[2])

        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        prediction = self.artifacts.model(input_data, training=False)
        # print(prediction)
        prediction_normed = softmax(prediction[0].numpy(), -1)
        prediction_idx = np.argmax(prediction_normed, -1).tolist()
        prediction_confidence = np.max(prediction_normed, -1).tolist()
        ret = []
        for idx, confidence in zip(prediction_idx, prediction_confidence):
            cur_label = self.decode_label_idx(idx)
            new_ret = {
                "label": cur_label,
                "confidence": confidence
            }
            ret.append(new_ret)

        return ret

    @api(JsonHandler, mb_max_latency=3000, mb_max_batch_size=20, batch=True)
    def multi_label_predict(self, parsed_json):
        logger.info(parsed_json)
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": []
        }
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(itm for itm in list(map(self.preprocessing, parsed_json)) if itm[0] is not None)))
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
            threshold = parsed_json[0].get("threshold", 0.5)
        else:  # expecting type(parsed_json) == dict:
            raise NotImplementedError
            # pre_input_data = self.preprocessing(parsed_json)
            # if pre_input_data[0] is None:
            #     return []
            # input_data['input_ids'].append(pre_input_data[0])
            # input_data['token_type_ids'].append(pre_input_data[1])
            # input_data['attention_mask'].append(pre_input_data[2])
            # threshold = parsed_json.get("threshold", 0.5)

        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        prediction = self.artifacts.model(input_data, training=False)
        prediction_normed = expit(prediction[0].numpy())

        ret = []

        for i, logits in enumerate(prediction_normed):
            logtit_idx = np.where(logits > threshold)[0].tolist()
            if len(logtit_idx) == 0:
                logtit_idx = [np.argmax(logits, -1)]

            one_ret = {"labels": [], "confidences": []}
            for j in logtit_idx:
                one_ret["labels"].append(self.decode_label_idx(j))
                one_ret["confidences"].append(float(prediction_normed[i, j]))

            ret.append(one_ret)

        return ret

    # curl -i \
    # --header "Content-Type: application/json" \
    # --request POST \
    # --data '{"text": ["空腹可以喝红茶吗", "因人而异，不能一概而论。"]}' \
    # http://127.0.0.1:5000/label_predict

    # bentoml run BertTextClassificationService:latest label_predict --input '{"text": ["空腹可以喝红茶吗", "因人而异，不能一概而论。"]}'