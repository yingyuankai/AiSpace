# # -*- coding: utf-8 -*-
# # @Time    : 2020-06-03 15:51
# # @Author  : yingyuankai
# # @Email   : yingyuankai@aliyun.com
# # @File    : test_gpt2.py
#
import unittest
import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import build_model


class TestGpt2(unittest.TestCase):
    def test_gpt2_checkpoint(self):
        # hparam = Hparams()
        # hparam.load_from_config_file('/search/odin/yyk/workspace/AiSpace/configs/pretrain/gpt.yml')
        # hparam.load_from_config_file('/search/data1/yyk/workspace/AiSpace/configs/glue_zh/tnews.yml')
        # hparam.load_from_config_file('/search/data1/yyk/workspace/AiSpace/configs/glue_zh/cmrc2018.yml')
        # hparam.stand_by()

        model_path = "/search/odin/yyk/data/pretrained/gpt/cpm-lm-tf2_v2"
        network = tf.keras.models.load_model(model_path)

        # ckpt_vars = [itm for itm in tf.train.list_variables(ckpt) if itm[0].find('adam') == -1]
        # ckpt_vars = [itm for itm in tf.train.list_variables(hparam.pretrained.model_path) if itm[0].find('adam') == -1]

        # model, (losses, loss_weights), metrics, optimizer = build_model(hparam)
        # model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        #
        # model_vars = model.trainable_variables

        print()