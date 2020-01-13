# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 15:38
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tnew_transformer.py


import os
from tqdm import tqdm
import json
from .base_transformer import BaseTransformer
from aispace.datasets import BaseTokenizer
from aispace.utils.io_utils import json_dumps

__all__ = [
    "TnewsTransformer"
]


@BaseTransformer.register("glue_zh/tnews")
class TnewsTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(TnewsTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

        # json dir
        self.json_dir = os.path.join(kwargs.get("data_dir", self._hparams.dataset.data_dir), "json")

    def transform(self, data_path, split="train"):
        output_path_base = os.path.join(os.path.dirname(data_path), "json")
        if not os.path.exists(output_path_base):
            os.makedirs(output_path_base)
        output_path = os.path.join(output_path_base, f"{split}.json")
        with open(data_path, "r", encoding="utf8") as inf:
            with open(output_path, "w", encoding="utf8") as ouf:
                for line in tqdm(inf):
                    if not line: continue
                    line = line.strip()
                    if len(line) == 0: continue
                    line_json = json.loads(line)
                    sentence = line_json.get("sentence", "").strip()
                    if len(sentence) == 0: continue
                    input_ids, token_type_ids, attention_mask = self.tokenizer.encode(sentence)
                    label = line_json.get("label_desc", "news_story")
                    item = {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attention_mask,
                        "label": label
                    }

                    new_line = f"{json_dumps(item)}\n"
                    ouf.write(new_line)
        return output_path