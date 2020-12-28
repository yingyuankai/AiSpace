# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 15:38
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tnew_transformer.py


import os
from tqdm import tqdm
import json
import logging
import numpy as np
# import hanlp
import pickle
from random import random, randrange
from pathlib import Path
from .base_transformer import BaseTransformer
from aispace.datasets import BaseTokenizer
from aispace.utils.io_utils import json_dumps
from aispace.utils.file_utils import default_download_dir, maybe_create_dir
from aispace.utils.io_utils import maybe_download, load_from_file

__all__ = [
    "GovTitleTriggerTransformer",
    "GovTitleRoleTransformer"
]

logger = logging.getLogger(__name__)


@BaseTransformer.register("gov_title/trigger")
class GovTitleTriggerTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(GovTitleTriggerTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        with open(data_path, "r") as inf:
            for idx, line in enumerate(inf):
                json_obj = json.loads(line)
                context = json_obj['context']
                titles: list = json_obj['titles']
                titles.sort(key=lambda s: s['span_start'])

                tokens = []
                labels = []
                pre_idx = 0
                for title in titles:
                    ss, se = title['span_start'], title['span_end']

                    pre_str = context[pre_idx: ss]
                    pre_tokens = self.tokenizer.tokenize(pre_str)
                    tokens.extend(pre_tokens)
                    labels.extend(["O"] * len(pre_tokens))

                    cur_str = title['text']
                    cur_tokens = self.tokenizer.tokenize(cur_str)
                    tokens.extend(cur_tokens)
                    labels.extend(["B"] + ["I"] * (len(cur_tokens) - 1))

                    pre_idx = se

                pre_str = context[pre_idx:]
                pre_tokens = self.tokenizer.tokenize(pre_str)
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                output = self.tokenizer.encode(tokens)

                labels = labels[: self.tokenizer.max_len - 2]
                labels = ["O"] + labels + ['O'] * (self.tokenizer.max_len - len(labels))

                feature = {
                    "input_ids": output['input_ids'],
                    "token_type_ids": output['segment_ids'],
                    "attention_mask": output['input_mask'],
                    "label": labels,
                }

                if idx == 0:
                    print(feature)
                yield feature


@BaseTransformer.register("gov_title/role")
class GovTitleRoleTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(GovTitleRoleTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        with open(data_path, "r") as inf:
            for idx, line in enumerate(inf):
                json_obj = json.loads(line)
                trigger = json_obj['trigger']
                roles: list = json_obj['roles']
                status = json_obj['status']
                context = json_obj['context']
                roles.append(trigger)
                roles.sort(key=lambda s: s['span_start'])

                tokens = []
                labels = []
                pre_idx = 0
                trigger_span_start, trigger_span_end = 1, 1
                for role in roles:
                    ss, se = role['span_start'], role['span_end']
                    entity_type = role['entity_type']

                    pre_str = context[pre_idx: ss]
                    pre_tokens = self.tokenizer.tokenize(pre_str)
                    tokens.extend(pre_tokens)
                    labels.extend(["O"] * len(pre_tokens))

                    cur_str = role['text']
                    cur_tokens = self.tokenizer.tokenize(cur_str)
                    tokens.extend(cur_tokens)
                    if role['entity_type'] != "TITLE":
                        labels.extend([f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(cur_tokens) - 1))
                    else:
                        trigger_span_start = len(labels)
                        labels.extend(["O"] * len(cur_tokens))
                        trigger_span_end = len(labels)

                    pre_idx = se

                pre_str = context[pre_idx:]
                pre_tokens = self.tokenizer.tokenize(pre_str)
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                query = f"{status}{trigger['text']}"
                query_tokens = self.tokenizer.tokenize(query)

                if trigger_span_end > self.tokenizer.max_len - 3 - len(query_tokens):
                    continue

                tokens = tokens[: self.tokenizer.max_len - 3 - len(query_tokens)]
                labels = labels[: self.tokenizer.max_len - 3 - len(query_tokens)]
                labels = ["O"] + labels + ['O'] * (self.tokenizer.max_len - len(labels) - 1)

                position_ids = list(range(0, 1 + len(tokens) + 1 + 2)) + \
                               list(range(trigger_span_start, trigger_span_end)) + list(range(len(tokens) + len(query_tokens) + 2, self.tokenizer.max_len))
                output = self.tokenizer.encode(tokens, query_tokens)

                feature = {
                    "input_ids": output['input_ids'],
                    "token_type_ids": output['segment_ids'],
                    "attention_mask": output['input_mask'],
                    "position_ids": position_ids,
                    "label": labels,
                }

                if idx == 0:
                    print(feature)
                yield feature