# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 20:11
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

import tensorflow as tf
import tensorflow_addons as tfa
from prettytable import PrettyTable

from aispace.utils.print_utils import print_boxed

__all__ = [
    "METRICS",
    "print_available"
]


METRICS = {
    "categorical_accuracy":
        lambda config: tf.keras.metrics.CategoricalAccuracy(**config),
    "sparse_categorical_accuracy":
        lambda config: tf.keras.metrics.SparseCategoricalAccuracy(**config),
    "sparse_categorical_crossentropy":
        lambda config: tf.keras.metrics.SparseCategoricalCrossentropy(**config),
    "f1_score":
        lambda config: tfa.metrics.F1Score(**config)
}

def print_available():
    table = PrettyTable(["NAME"])
    for key in METRICS:
        table.add_row([key])
    print()
    print(table)