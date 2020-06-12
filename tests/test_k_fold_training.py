# -*- coding: utf-8 -*-
# @Time    : 2020-06-11 20:03
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_k_fold_training.py


import unittest

from aispace.utils.hparams import Hparams
from aispace.trainer import k_fold_experiment


class TestKFoldTraining(unittest.TestCase):
    def test_dataset_split(self):
        hparams = Hparams()
        hparams.load_from_config_file("../configs/glue_zh/tnews.yml")
        hparams.stand_by()

        k_fold_experiment(hparams)
