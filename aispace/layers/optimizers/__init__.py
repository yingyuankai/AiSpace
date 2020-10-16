# -*- coding: utf-8 -*-
# @Time    : 2019-11-14 20:50
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

import tensorflow as tf
import tensorflow_addons as tfa
from prettytable import PrettyTable

from .adam_weight_decay_optimizer import create_awdwwu_optimizer
from .lr_multiplier import LRMultiplier
# from keras_lr_multiplier import LRMultiplier

__all__ = [
    "OPTIMIZERS",
    "OPTIMIZER_WRAPPER",
    "print_available"
]

OPTIMIZERS = {
    'sgd':
        lambda training_hparams: tf.keras.optimizers.SGD(
            learning_rate=training_hparams.learning_rate
        ),
    'adam':
        lambda training_hparams: tf.keras.optimizers.Adam(
            learning_rate=training_hparams.learning_rate, epsilon=1e-08, clipnorm=1.0),
    'adam_weight_decay_with_warm_up': create_awdwwu_optimizer,
    'radam':
        lambda training_hparams:
        tfa.optimizers.RectifiedAdam(training_hparams.learning_rate,
                                     total_steps=training_hparams.steps_per_epoch * training_hparams.max_epochs,
                                     warmup_proportion=min(training_hparams.warmup_factor,
                                                           float(training_hparams.steps_per_epoch) / (
                                                                   training_hparams.steps_per_epoch * training_hparams.max_epochs)),
                                     weight_decay=0.01)
}

OPTIMIZER_WRAPPER = {
    'swa':
        lambda opt, training_hparams:
        tfa.optimizers.SWA(opt, start_averaging=training_hparams.steps_per_epoch *
                                                training_hparams.optimizer_wrappers.get('swa').config.start_epoch,
                           average_period=training_hparams.steps_per_epoch // 10),
    # TODO
    # 'lr_multiplier': lambda opt, training_hparams: LRMultiplier(opt, multipliers=training_hparams.optimizer_wrappers.get('lr_multiplier').config.multipliers)
}


def print_available():
    table = PrettyTable(["NAME"])
    for key in OPTIMIZERS:
        table.add_row([key])

    for key in OPTIMIZER_WRAPPER:
        table.add_row([key])
    print()
    print(table)
