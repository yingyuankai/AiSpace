# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:01
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_glue.py

import tensorflow_datasets as tfds
import unittest

from tensorflow_datasets.core.download import DownloadConfig

from aispace.datasets import *
from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import load_dataset


class TestGlue(unittest.TestCase):
    def test_glue_load(self):
        hparams = Hparams()
        checksum_dir = "../aispace/datasets/url_checksums"
        tfds.download.add_checksums_dir(checksum_dir)
        # download_config = DownloadConfig(register_checksums=False)
        tnews = tfds.load("glue_zh/tnews",
                          # data_dir="/search/data1/yyk/data/datasets/glue_zh",
                          data_dir="../data/glue_zh",
                          builder_kwargs={'hparams': hparams},
                          # download_and_prepare_kwargs={'download_config': download_config}
                          )
        for item in tnews['test']:
            print()
            # print(item["context"].numpy().decode("utf8"))
            # print(item["question"].numpy().decode("utf8"))
            print(item["sentence"].numpy().decode("utf8"))
            print(item)
            break
        print()

    def test_tnews(self):
        hparam = Hparams()
        hparam.load_from_config_file('../configs/glue_zh/tnews.yml')
        hparam.stand_by()

        train_data, dev_data, test_data, data_info = load_dataset(hparam)
        for item in train_data.take(1):
            print()
            # print(item["context"].numpy().decode("utf8"))
            # print(item["question"].numpy().decode("utf8"))
            # print(item["sentence"].numpy().decode("utf8"))
            print(item)
            break
        print()
