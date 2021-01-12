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
    "IdiomGeneratorTransformer",
]

logger = logging.getLogger(__name__)


@BaseTransformer.register("idiom/idiom_generator")
class IdiomGeneratorTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(IdiomGeneratorTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

    def transform(self, data_path, split="train"):
        with open(data_path, "r") as inf:
            for idx, line in enumerate(inf):
                json_obj = json.loads(line)
                gushi = json_obj['gushi']
                chenyu = json_obj['chenyu']

                gushi = gushi.replace(chenyu, "")

                gushi_tokens = self.tokenizer.tokenize(gushi)
                chenyu_tokens = self.tokenizer.tokenize(chenyu)

                gushi_tokens = gushi_tokens[: self.tokenizer.max_len - len(chenyu_tokens) - 1]

                tokens = gushi_tokens + [self.tokenizer.vocab.sep_token] + chenyu_tokens + [self.tokenizer.vocab.eod_token]

                input_tokens = tokens[:-1]
                label_tokens = []
                for token in tokens[1:]:
                    if token in self.tokenizer.vocab.token_to_id:
                        label_tokens.append(token)
                    else:
                        label_tokens.append(self.tokenizer.vocab.unk_token)

                attention_mask = [1] * len(input_tokens) + [0] * (self.tokenizer.max_len - len(input_tokens))

                input_tokens += [self.tokenizer.vocab.pad_token] * (self.tokenizer.max_len - len(input_tokens))
                label_tokens += [self.tokenizer.vocab.pad_token] * (self.tokenizer.max_len - len(input_tokens))

                encode_output = self.tokenizer.encode(input_tokens)

                feature = {
                    "input_ids": encode_output["input_ids"],
                    "attention_mask": attention_mask,
                    "label": label_tokens,
                }

                if idx == 0:
                    print(feature)
                    print(len(feature['label']))
                yield feature