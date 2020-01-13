# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 15:36
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : base_transformer.py

from aispace.utils.registry import Registry


class BaseTransformer(Registry):
    def __init__(self, hparams, **kwargs):
        self._hparams = hparams

    def transform(self, *args, **kwargs):
        raise NotImplementedError