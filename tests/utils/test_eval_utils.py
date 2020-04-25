# !/usr/bin/env python
# coding=utf-8
# @Time    : 2020/4/25 19:44
# @Author  : yingyuankai@aliyun.com
# @File    : test_eval_utils.py

import unittest
from aispace.utils.hparams import Hparams
from aispace.utils.eval_utils import evaluation


class TestEvalUtils(unittest.TestCase):
    def test_eval(self):
        hparams = Hparams()
        hparams.load_from_config_file("../../configs/2020_LSTC/DuEE_trigger_as_classifier.yml")
        hparams.stand_by()
        evaluation(hparams)
