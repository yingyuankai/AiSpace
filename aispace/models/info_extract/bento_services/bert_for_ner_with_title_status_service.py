# -*- coding: utf-8 -*-
# @Time    : 2019-12-03 20:41
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_for_sequence_classification.py

__all__ = [
    "BertNerWithTitleStatusService"
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
"""roles extraction with title and its status"""

@artifacts([
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('tokenizer'),
        PickleArtifact("hparams"),
    ])
@env(auto_pip_dependencies=True)
class BertNerWithTitleStatusService(BentoService):

    def preprocessing(self, itm):
        context = itm.get("text")
        span_start, span_end = itm.get("trigger_start"), itm.get("trigger_end")
        status = itm.get('status')
        trigger = itm.get("trigger")

        tokens = []
        cur_tokens = self.artifacts.tokenizer.tokenize(context[0: span_start])
        tokens.extend(cur_tokens)
        trigger_token_start_idx = len(tokens)
        cur_tokens = self.artifacts.tokenizer.tokenize(trigger)
        tokens.extend(cur_tokens)
        trigger_token_end_idx = len(tokens)

        cur_tokens = self.artifacts.tokenizer.tokenize(context[span_end:])
        tokens.extend(cur_tokens)

        query = f"{status}{trigger}"
        query_tokens = self.artifacts.tokenizer.tokenize(query)

        tokens = tokens[: self.artifacts.tokenizer.max_len - 3 - len(query_tokens)]

        position_ids = list(range(0, 1 + len(tokens) + 1 + 2)) + \
                       list(range(trigger_token_start_idx, trigger_token_end_idx)) + list(
            range(len(tokens) + len(query_tokens) + 2, self.artifacts.tokenizer.max_len))
        output = self.artifacts.tokenizer.encode(tokens, query_tokens)

        input_ids, token_type_ids, attention_mask = output['input_ids'], output['segment_ids'], output['input_mask']
        return input_ids, token_type_ids, attention_mask, position_ids, context

    def _align_raw_text(self, tags, raw_tokens, align_mapping):
        new_tokens, new_tags = [], []
        i, j = 0, 0
        while i <= j < min(len(tags), len(raw_tokens)):
            if align_mapping[i] == align_mapping[j]:
                j += 1
            else:
                new_tokens.append(raw_tokens[align_mapping[i]])
                new_tags.append(tags[i])
                i = j
        return new_tokens, new_tags

    def postprocessing(self, tokens, tags):
        assert len(tokens) == len(tags), \
            ValueError(f'tokens length {len(tokens)} is not equal to tags length {len(tags)}!')
        ret = {"passage": "", "entities": []}
        i, j, t = 0, 0, 0
        text = ""
        cur_tag = ""
        while i < len(tags) and len(tags) > j >= i:
            if not tags[i].startswith("B-"):
                i += 1
                j = i
            elif i == j and tags[i].startswith("B-"):
                cur_tag = tags[i][2:]
                j += 1
            elif tags[j] == f"I-{cur_tag}":
                j += 1
            elif tags[j] == 'O' or tags[j][2:] != cur_tag or tags[j].startswith("B-"):
                pre_tokens = tokens[t: i]
                entity_tokens = tokens[i: j]
                pre_text = self.artifacts.tokenizer.detokenizer(pre_tokens)
                entity_text = self.artifacts.tokenizer.detokenizer(entity_tokens)
                text += pre_text
                new_entity = {
                    "text": entity_text,
                    "span_start": len(text),
                    "span_end": len(text) + len(entity_text),
                    "tag": cur_tag
                }
                text += entity_text
                t = i = j
                ret["entities"].append(new_entity)
                cur_tag = ""
        if i != t:
            pre_tokens = tokens[t: i]
            pre_text = self.artifacts.tokenizer.detokenizer(pre_tokens)
            text += pre_text
        if i != j:
            entity_tokens = tokens[i: j]
            entity_text = self.artifacts.tokenizer.detokenizer(entity_tokens)
            new_entity = {
                "text": entity_text,
                "span_start": len(text),
                "span_end": len(text) + len(entity_text),
                "tag": cur_tag
            }
            text += entity_text
            ret["entities"].append(new_entity)
        ret["passage"] = text
        return ret

    def decode_label_idx(self, idx):
        return self.artifacts.hparams.dataset.outputs[0].labels[idx]

    def decode_token_idx(self, idx):
        return self.artifacts.tokenizer.decode(idx)

    @api(JsonHandler)
    def ner_predict(self, parsed_json):
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": [], "position_ids": []
        }
        seq_length = []
        passages = []
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(map(self.preprocessing, parsed_json))))
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
            input_data['position_ids'].extend(pre_input_data[3])
            seq_length.extend(list(map(sum, pre_input_data[2])))
            passages.extend(pre_input_data[-1])
        else:  # expecting type(parsed_json) == dict:
            pre_input_data = self.preprocessing(parsed_json)
            label_mask = parsed_json.get("label_mask")
            input_data['input_ids'].append(pre_input_data[0])
            input_data['token_type_ids'].append(pre_input_data[1])
            input_data['attention_mask'].append(pre_input_data[2])
            input_data['position_ids'].append(pre_input_data[3])
            input_data['label_mask'].append(label_mask)
            seq_length.append(sum(pre_input_data[2]))
            passages.append(pre_input_data[-1])
        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        input_data['position_ids'] = tf.constant(input_data['position_ids'], name="position_ids")
        prediction = self.artifacts.model(input_data, training=False)
        prediction_idx = np.argmax(prediction, -1).tolist()
        ret = []
        for idx, token_ids, seq_len, passage in zip(prediction_idx, input_data["input_ids"].numpy().tolist(), seq_length, passages):
            cur_labels = list(map(self.decode_label_idx, idx[1: seq_len - 1]))
            # cur_tokens = self.decode_token_idx(token_ids[1:seq_len - 1])
            raw_tokens, _, align_mapping = self.artifacts.tokenizer.tokenize(passage, True)
            cur_tokens, cur_labels = self._align_raw_text(cur_labels, raw_tokens, align_mapping)
            new_ret = self.postprocessing(cur_tokens, cur_labels)
            ret.append(new_ret)
        return ret


# {
#     "text": "2006.06-2007.02雨山区委副书记，区人民政府代区长；",
#     "status": "曾任",
#     "trigger": "副书记",
#     "trigger_start": 19,
#     "trigger_end": 21
# }