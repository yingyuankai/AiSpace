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
from tqdm import tqdm
from aispace.datasets import *
from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import load_dataset


class TestGlue(unittest.TestCase):
    def test_glue_load(self):
        hparams = Hparams()
        hparams.load_from_config_file("../configs/qa/dureader_robust.yml")
        hparams.stand_by()
        # checksum_dir = "../aispace/datasets/url_checksums"
        # tfds.download.add_checksums_dir(checksum_dir)
        # download_config = DownloadConfig(register_checksums=True)
        # dureader = tfds.load("dureader/robust",
        #                   # data_dir="/search/data1/yyk/data/datasets/glue_zh",
        #                   data_dir="../data/dureader",
        #                   builder_kwargs={'hparams': hparams},
        #                   download_and_prepare_kwargs={'download_config': download_config}
        #                   )

        # train_dataset, dev_dataset, dataset_info = next(load_dataset(hparams, ret_test=False))
        test_dataset = next(load_dataset(hparams, ret_train=True, ret_dev=True, ret_test=True, ret_info=True))[0]

        total, zero = 0, 0
        for itm in tqdm(test_dataset):
            # tt = itm[0]['input_ids'].numpy().tolist()
            # print(itm[0]['p_mask'].numpy().tolist())
            # print(itm[0]['start_position'].numpy().tolist())
            # print(itm[0]['end_position'].numpy().tolist())
            # print(tt)
            # break
            total += 1
            # zero += len([t for t in tt if t == 0])
        # print()
        # print(f"{zero}, {total}, {zero / float(total)}")
        print(total)

    def test_tttt(self):
        import math

        print(math.perm(3, 2))

# python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule train_and_eval \
#    --config_name tnews \
#    --config_dir ./configs/glue_zh \
#    --gpus 0 1 2 3