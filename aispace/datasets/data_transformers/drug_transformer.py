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
    "DrugTransformer"
]

logger = logging.getLogger(__name__)


@BaseTransformer.register("drug")
class DrugTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(DrugTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        limit = 1000000
        with open(data_path, "r") as inf:
            for idx, line in enumerate(inf):
                if idx >= limit:
                    break
                json_obj = json.loads(line)
                context = json_obj['context']
                entities: list = json_obj['entities']
                entities.sort(key=lambda s: s['span_start'])

                tokens = []
                labels = []
                pre_idx = 0
                for entity in entities:
                    ss, se = entity['span_start'], entity['span_end']
                    entity_type = entity['entity_type']
                    pre_str = context[pre_idx: ss]
                    pre_tokens = self.tokenizer.tokenize(pre_str)
                    tokens.extend(pre_tokens)
                    labels.extend(["O"] * len(pre_tokens))

                    cur_str = entity['text']
                    cur_tokens = self.tokenizer.tokenize(cur_str)
                    tokens.extend(cur_tokens)
                    labels.extend([f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(cur_tokens) - 1))

                    pre_idx = se

                pre_str = context[pre_idx:]
                pre_tokens = self.tokenizer.tokenize(pre_str)
                tokens.extend(pre_tokens)
                labels.extend(["O"] * len(pre_tokens))

                output = self.tokenizer.encode(tokens)

                if self._hparams.dataset.tokenizer.name != "gpt_tokenizer":
                    labels = ["O"] + labels[: self.tokenizer.max_len - 2]
                    labels = labels + ['O'] * (self.tokenizer.max_len - len(labels))
                else:
                    labels = labels[: self.tokenizer.max_len]
                    labels = labels + ['O'] * (self.tokenizer.max_len - len(labels))

                feature = {
                    "input_ids": output['input_ids'],
                    "token_type_ids": output['segment_ids'],
                    "attention_mask": output['input_mask'],
                    "label": labels,
                }

                if idx == 0:
                    print(feature)
                    print(len(feature['label']))
                yield feature