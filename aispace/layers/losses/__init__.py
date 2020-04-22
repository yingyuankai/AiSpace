# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 20:39
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py

__all__ = [
    "LOSSES"
]

from prettytable import PrettyTable
import tensorflow as tf


from .focal_losses import \
    focal_loss_softmax, focal_loss_sigmoid_v1, focal_loss_sigmoid_v2

from .dice_loss import dice_loss, cce_dice_loss

LOSSES = {
    "sparse_categorical_crossentropy": lambda loss_config:
    tf.keras.losses.SparseCategoricalCrossentropy(**loss_config),
    'focal_loss_softmax': lambda loss_config: focal_loss_softmax(**loss_config),
    'focal_loss_sigmoid_v1': lambda loss_config: focal_loss_sigmoid_v1(**loss_config),
    'focal_loss_sigmoid_v2': lambda loss_config: focal_loss_sigmoid_v2(**loss_config),
    'dice_loss': lambda loss_config: dice_loss(**loss_config),
    "cce_dice_loss": lambda loss_config: cce_dice_loss(**loss_config)
}


def print_available():
    table = PrettyTable(["NAME"])
    for key in LOSSES:
        table.add_row([key])
    print()
    print(table)