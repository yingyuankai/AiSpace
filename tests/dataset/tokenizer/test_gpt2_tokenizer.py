# # -*- coding: utf-8 -*-
# # @Time    : 2020-06-02 16:17
# # @Author  : yingyuankai
# # @Email   : yingyuankai@aliyun.com
# # @File    : test_xlnet_tokenizer.py


import unittest

from aispace.utils.hparams import Hparams

from aispace.datasets.tokenizer import CPMTokenizer


class TestXlnetTokenizer(unittest.TestCase):

    def test_init(self):
        hparams = Hparams()
        hparams.load_from_config_file("../../../configs/custom/test_gpt2.yml")
        hparams.stand_by()
        tokenizer = CPMTokenizer(hparams.dataset.tokenizer)

        a = "这两天，XLNet貌似也引起了NLP圈的极大关注，从实验数据看，在某些场景下，确实XLNet相对Bert有很大幅度的提升。"
        b = "就像我们之前说的，感觉Bert打开两阶段模式的魔法盒开关后，在这条路上，会有越来越多的同行者，而XLNet就是其中比较引人注目的一位"

        res = tokenizer.encode(a, b)
        print(res)