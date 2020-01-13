# -*- coding: utf-8 -*-
# @Time    : 2019-11-15 15:40
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

import tensorflow as tf

from swa.tfkeras import SWA

__all__ = [
    "CALLBACKS"
]

CALLBACKS = {
    'early_stopping': tf.keras.callbacks.EarlyStopping,
    'checkpoint': tf.keras.callbacks.ModelCheckpoint,
    'tensorboard': tf.keras.callbacks.TensorBoard,
    'swa': SWA
}