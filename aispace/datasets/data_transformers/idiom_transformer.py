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


                feature = {
                    "input_ids": output['input_ids'],
                    "label": labels,
                }

                if idx == 0:
                    print(feature)
                    print(len(feature['label']))
                yield feature