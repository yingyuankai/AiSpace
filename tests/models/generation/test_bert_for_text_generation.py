# -*- coding: utf-8 -*-
# @Time    : 1/12/21 2:52 PM
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_bert_for_text_generation.py


import unittest
import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import build_model
from aispace.datasets.tokenizer import CPMTokenizer


class TestGptAdapter(unittest.TestCase):
    def test_process(self):
        hparam = Hparams()
        hparam.load_from_config_file('/search/odin/yyk/workspace/AiSpace/configs/custom/test_gpt2.yml')
        hparam.stand_by()
        model, (losses, loss_weights), metrics, optimizer = build_model(hparam)
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

        tokenizer = CPMTokenizer(hparam.dataset.tokenizer)

        input = "我爱你是因为你的美丽的心灵和外表"

        input_encoded = tokenizer.encode(input)

        input_ids = tf.constant([input_encoded['input_ids']], dtype=tf.int32)
        input_dict = {
            "input_ids": input_ids
        }
        # output = model(input_dict)
        output = model.generate(input_ids, **hparam.generation_attributes)

        print(input_encoded)
        output = tokenizer.decode(output.numpy().reshape([-1]).tolist())
        print(output)
