# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 20:50
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

import tensorflow as tf
from prettytable import PrettyTable

from .adam_weight_decay_optimizer import create_awdwwu_optimizer

__all__ = [
    "OPTIMIZERS",
    "print_available"
]

OPTIMIZERS = {
    'sgd': lambda training_hparams: tf.keras.optimizers.SGD(
        learning_rate=training_hparams.learning_rate
    ),
    'adam': lambda training_hparams: tf.keras.optimizers.Adam(
        learning_rate=training_hparams.learning_rate, epsilon=1e-08, clipnorm=1.0),
    'adam_weight_decay_with_warm_up': create_awdwwu_optimizer
}


def print_available():
    table = PrettyTable(["NAME"])
    for key in OPTIMIZERS:
        table.add_row([key])
    print()
    print(table)