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
        hparams.load_from_config_file("../../configs/glue_zh/tnews_k_fold.yml")
        hparams.stand_by()
        ckpts = [
            "../../save/test_textcnn_for_classification_119_14/k_fold/1/model_saved/model",
            "../../save/test_textcnn_for_classification_119_14/k_fold/2/model_saved/model",
        ]
        evaluation(hparams, checkpoints=ckpts)
