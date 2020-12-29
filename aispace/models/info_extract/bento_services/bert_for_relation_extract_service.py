# -*- coding: utf-8 -*-
# @Time    : 2019-12-03 20:41
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_sequence_classification.py

__all__ = [
    "BertRelationClassificationService"
]

import os, sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" * 4)))

from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TensorflowSavedModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

import numpy as np
from scipy.special import softmax

from aispace.datasets.tokenizer import BertTokenizer
from aispace.utils.hparams import Hparams


@artifacts([
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('tokenizer'),
        PickleArtifact("hparams"),
    ])
@env(auto_pip_dependencies=True)
class BertRelationClassificationService(BentoService):

    def preprocessing(self, one_inst):
        text_str = one_inst['text']
        entity1_span = one_inst["entity1_span"] + ["[ENTITY1_START]", "[ENTITY1_END]"]
        entity2_span = one_inst["entity2_span"] + ["[ENTITY2_START]", "[ENTITY2_END]"]
        entities = [entity1_span, entity2_span]
        entities.sort(key=lambda s: s[0])
        tokens = [self.artifacts.tokenizer.vocab.cls_token]
        entity_span_starts, entity_span_ends = [0] * 2, [0] * 2
        pre_idx = 0
        for entity_start, entity_end, entity_start_marker, entity_end_marker in entities:
            if pre_idx < entity_start:
                sub_text_tokens = self.artifacts.tokenizer.tokenize(text_str[pre_idx: entity_start])
                tokens.extend(sub_text_tokens)
            if entity_start_marker.startswith("[ENTITY1"):
                entity_span_starts[0] = len(tokens)
            else:
                entity_span_starts[1] = len(tokens)
            tokens.append(entity_start_marker)
            entity_tokens = self.artifacts.tokenizer.tokenize(text_str[entity_start:entity_end + 1])
            tokens.extend(entity_tokens)
            if entity_start_marker.startswith("[ENTITY1"):
                entity_span_ends[0] = len(tokens)
            else:
                entity_span_ends[1] = len(tokens)
            tokens.append(entity_end_marker)
            pre_idx = entity_end + 1

        if pre_idx < len(text_str):
            sub_text_tokens = self.artifacts.tokenizer.tokenize(text_str[pre_idx:])
            tokens.extend(sub_text_tokens)
        tokens = tokens[:153]
        tokens.append(self.artifacts.tokenizer.vocab.sep_token)
        input_ids = self.artifacts.tokenizer.vocab.transformer(tokens) + [0] * (154 - len(tokens))
        token_type_ids = [0] * 154
        attention_mask = [1] * len(tokens) + [0] * (154 - len(tokens))
        return input_ids, token_type_ids, attention_mask, entity_span_starts, entity_span_ends

    def decode_label_idx(self, idx):
        return self.artifacts.hparams.dataset.outputs[0].labels[idx]

    @api(JsonHandler)
    def relation_predict(self, parsed_json):
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": [], "entity_span_start": [], "entity_span_end": []
        }
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(map(self.preprocessing, parsed_json))))
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
            input_data['entity_span_start'].extend(pre_input_data[3])
            input_data['entity_span_end'].extend(pre_input_data[4])
        else:  # expecting type(parsed_json) == dict:
            pre_input_data = self.preprocessing(parsed_json)
            input_data['input_ids'].append(pre_input_data[0])
            input_data['token_type_ids'].append(pre_input_data[1])
            input_data['attention_mask'].append(pre_input_data[2])
            input_data['entity_span_start'].append(pre_input_data[3])
            input_data['entity_span_end'].append(pre_input_data[4])
        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        input_data['entity_span_start'] = tf.constant(input_data['entity_span_start'], name="entity_span_start")
        input_data['entity_span_end'] = tf.constant(input_data['entity_span_end'], name="entity_span_end")
        prediction = self.artifacts.model(input_data, training=False)
        prediction_normed = softmax(prediction.numpy(), -1)
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

    # curl -i \
    # --header "Content-Type: application/json" \
    # --request POST \
    # --data '{"text": "任命李炳丰同志为陆丰市农业农村局局长，免去其陆丰市林业局局长职务。", "entity1_span": [2, 4], "entity2_span": [28, 29]}' \
    # http://localhost:5002/relation_predict
