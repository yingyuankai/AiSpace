# -*- coding: utf-8 -*-
# @Time    : 2019-11-15 15:40
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

import tensorflow as tf

from prettytable import PrettyTable

from aispace.layers.callbacks.lr_finder import LRFinder
from aispace.layers.callbacks.qa_evaluators import EvaluatorForQaWithImpossible, EvaluatorForQaSimple

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
    'lr_finder': lambda config: LRFinder(**config),
    'evaluator_for_qa_with_impossible':
        lambda config: EvaluatorForQaWithImpossible(**config),
    'evaluator_for_qa_simple':
        lambda config: EvaluatorForQaSimple(**config)
}


def print_available():
    table = PrettyTable(["NAME"])
    for key in CALLBACKS:
        table.add_row([key])
    print()
    print(table)