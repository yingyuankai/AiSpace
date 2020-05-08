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
import tensorflow_addons as tfa
# from .focal_losses import \
#     focal_loss_softmax, focal_loss_sigmoid_v1, focal_loss_sigmoid_v2

from .dice_loss import DceLoss, CceDceLoss
from .cross_entropy_loss import SigmoidCrossEntropy
from .focal_loss import SparseSoftmaxFocalCrossEntropy

LOSSES = {
    "categorical_crossentropy":
        lambda loss_config: tf.keras.losses.CategoricalCrossentropy(**loss_config),
    "sparse_categorical_crossentropy":
        lambda loss_config: tf.keras.losses.SparseCategoricalCrossentropy(**loss_config),
    'sigmoid_focal_crossentropy':
        lambda loss_config: tfa.losses.SigmoidFocalCrossEntropy(**loss_config),
    "sparse_softmax_focal_crossentropy":
        lambda loss_config: SparseSoftmaxFocalCrossEntropy(**loss_config),
    'sigmoid_cross_entropy':
        lambda loss_config: SigmoidCrossEntropy(**loss_config),
    'dce_loss':
        lambda loss_config: DceLoss(**loss_config),
    "cce_dce_loss":
        lambda loss_config: CceDceLoss(**loss_config)
}


def print_available():
    table = PrettyTable(["NAME"])
    for key in LOSSES:
        table.add_row([key])
    print()
    print(table)