# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:01
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_glue.py

import os, sys
import tensorflow_datasets as tfds
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow_datasets.core.download import DownloadConfig

from aispace.datasets import *
from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import load_dataset


class TestLSTC(unittest.TestCase):
    def test_lstc_load(self):
        hparams = Hparams()
        hparams.load_from_config_file("../configs/2020_LSTC/DuEE_role2.yml")
        hparams.stand_by()
        checksum_dir = "../aispace/datasets/url_checksums"
        tfds.download.add_checksums_dir(checksum_dir)
        download_config = DownloadConfig(register_checksums=True)
        tnews = tfds.load("lstc_2020/DuEE_role",
                          # data_dir="/search/data1/yyk/data/datasets/glue_zh",
                          data_dir="../data",
                          builder_kwargs={'hparams': hparams},
                          download_and_prepare_kwargs={'download_config': download_config}
                          )

        tokenizer = BertTokenizer(hparams.dataset.tokenizer)
        for itm in tnews["train"]:
            # for k, v in itm.items():
            #     if v.shape[0] == 152:
            #         print(itm)
            #         break
            print(itm)
            print(tokenizer.decode(itm["input_ids"].numpy().tolist()))
            break

# python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule train_and_eval \
#    --config_name tnews \
#    --config_dir ./configs/glue_zh \
#    --gpus 0 1 2 3