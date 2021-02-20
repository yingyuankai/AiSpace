# -*- coding: utf-8 -*-
# @Time    : 1/6/21 3:11 PM
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_gpt2_adapter.py

import unittest

from aispace.layers.adapters import tf_huggingface_gpt2_adapter
from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import build_model


class TestGptAdapter(unittest.TestCase):
    def test_process(self):
        hparam = Hparams()
        hparam.load_from_config_file('/search/odin/yyk/workspace/AiSpace/configs/custom/test_gpt2.yml')
        hparam.stand_by()
        model, (losses, loss_weights), metrics, optimizer = build_model(hparam)
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        model_vars = model.trainable_variables
        model_path = "/search/odin/yyk/data/pretrained/gpt/cpm-lm-tf2_v2"
        tf_huggingface_gpt2_adapter(model_vars, model_path)