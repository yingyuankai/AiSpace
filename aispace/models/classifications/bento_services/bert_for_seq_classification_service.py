# -*- coding: utf-8 -*-
# @Time    : 2019-12-03 20:41
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_sequence_classification.py

__all__ = [
    "BertTextClassificationService"
]

import os, sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" * 4)))

from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TensorflowSavedModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

import numpy as np
from scipy.special import softmax, expit

from aispace.datasets.tokenizer import BertTokenizer
from aispace.utils.hparams import Hparams


@artifacts([
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('tokenizer'),
        PickleArtifact("hparams"),
    ])
@env(pip_dependencies=['tensorflow-gpu==2.0.0', 'numpy==1.16', 'scipy==1.3.1', "tensorflow-datasets==1.3.0"])
class BertTextClassificationService(BentoService):

    def preprocessing(self, one_json):
        text_str = one_json["text"]
        input_ids, token_type_ids, attention_mask = self.artifacts.tokenizer.encode(text_str)
        return input_ids, token_type_ids, attention_mask

    def decode_label_idx(self, idx):
        return self.artifacts.hparams.dataset.outputs[0].labels[idx]

    @api(JsonHandler)
    def label_predict(self, parsed_json):
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": []
        }
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(map(self.preprocessing, parsed_json))))
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
        else:  # expecting type(parsed_json) == dict:
            pre_input_data = self.preprocessing(parsed_json)
            input_data['input_ids'].append(pre_input_data[0])
            input_data['token_type_ids'].append(pre_input_data[1])
            input_data['attention_mask'].append(pre_input_data[2])

        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        prediction = self.artifacts.model(input_data, training=False)
        prediction_normed = softmax(prediction[0].numpy(), -1)
        prediction_idx = np.argmax(prediction_normed, -1).tolist()
        prediction_confidence = np.max(prediction_normed, -1).tolist()
        ret = {
            "predictions": []
        }
        for idx, confidence in zip(prediction_idx, prediction_confidence):
            cur_label = self.decode_label_idx(idx)
            new_ret = {
                "label": cur_label,
                "confidence": confidence
            }
            ret["predictions"].append(new_ret)

        return ret

    @api(JsonHandler)
    def multi_label_predict(self, parsed_json):
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": []
        }
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(map(self.preprocessing, parsed_json))))
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
            threshold = parsed_json[0].get("threshold", 0.5)
        else:  # expecting type(parsed_json) == dict:
            pre_input_data = self.preprocessing(parsed_json)
            input_data['input_ids'].append(pre_input_data[0])
            input_data['token_type_ids'].append(pre_input_data[1])
            input_data['attention_mask'].append(pre_input_data[2])
            threshold = parsed_json.get("threshold", 0.5)

        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        prediction = self.artifacts.model(input_data, training=False)
        prediction_normed = expit(prediction[0].numpy())

        ret = {
            "predictions": []
        }

        for i, logits in enumerate(prediction_normed):
            logtit_idx = np.where(logits > threshold)[0].tolist()
            if len(logtit_idx) == 0:
                logtit_idx = [np.argmax(logits, -1)]

            one_ret = {"labels": [], "confidences": []}
            for j in logtit_idx:
                one_ret["labels"].append(self.decode_label_idx(j))
                one_ret["confidences"].append(float(prediction_normed[i, j]))

            ret["predictions"].append(one_ret)

        return ret

    # curl -i \
    # --header "Content-Type: application/json" \
    # --request POST \
    # --data '{"text": "泰安今早发生2.9级地震！靠近这个国家森林公园"}' \
    # http://127.0.0.1:5000/label_predict