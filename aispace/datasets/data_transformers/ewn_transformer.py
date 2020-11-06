# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 15:38
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tnew_transformer.py


import os
import logging
from tqdm import tqdm
import json
import numpy as np
from .base_transformer import BaseTransformer
from aispace.datasets import BaseTokenizer
from aispace.utils.io_utils import json_dumps
from aispace.utils.str_utils import preprocess_text

__all__ = [
    "EntityWithNationalityTransformer",
]

logger = logging.getLogger(__name__)


@BaseTransformer.register("entity_with_nationality")
class EntityWithNationalityTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(EntityWithNationalityTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

        # json dir
        self.json_dir = os.path.join(kwargs.get("data_dir", self._hparams.dataset.data_dir), "json")

    def transform(self, data_path, split="train"):
        # output_path_base = os.path.join(os.path.dirname(data_path), "json")
        # if not os.path.exists(output_path_base):
        #     os.makedirs(output_path_base)
        # output_path = os.path.join(output_path_base, f"{split}.json")
        with open(data_path, "r", encoding="utf8") as inf:
            for line in inf:
                if not line: continue
                line = line.strip()
                if len(line) == 0: continue
                line_json = json.loads(line)
                sentence = line_json.get("entity", "").strip()
                if len(sentence) == 0: continue
                encode_info = self.tokenizer.encode(sentence)
                input_ids, token_type_ids, attention_mask = \
                    encode_info['input_ids'], encode_info['segment_ids'], encode_info['input_mask']
                label = line_json.get("label")
                item = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "label": label
                }
                yield item
