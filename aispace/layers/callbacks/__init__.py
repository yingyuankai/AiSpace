# -*- coding: utf-8 -*-
# @Time    : 2019-11-15 15:40
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

import tensorflow as tf

from aispace.layers.callbacks.lr_finder import LRFinder

__all__ = [
    "CALLBACKS"
]

CALLBACKS = {
    'early_stopping':
        lambda config: tf.keras.callbacks.EarlyStopping(**config),
    'checkpoint':
        lambda config: tf.keras.callbacks.ModelCheckpoint(**config),
    'tensorboard':
        lambda config: tf.keras.callbacks.TensorBoard(**config),
    'lr_finder': lambda config: LRFinder(**config)
}