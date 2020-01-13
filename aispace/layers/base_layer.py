# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-05 10:27
# @Author  : yingyuankai@aliyun.com
# @File    : base_model.py

__all__ = [
    "BaseLayer"
]

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.registry import Registry


class BaseLayer(tf.keras.layers.Layer, Registry):
    __metaclass__ = ABCMeta

    def __init__(self, hparams: Hparams, **kwargs):
        super(BaseLayer, self).__init__(**kwargs)
        self._hparams = hparams

