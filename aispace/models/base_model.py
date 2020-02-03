# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-05 10:27
# @Author  : yingyuankai@aliyun.com
# @File    : base_model.py

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.registry import Registry


__all__ = [
    "BaseModel"
]


class BaseModel(tf.keras.Model, Registry):
    __metaclass__ = ABCMeta

    def __init__(self, hparams: Hparams, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self._hparams = hparams

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @abstractmethod
    def deploy(self):
        raise NotImplementedError