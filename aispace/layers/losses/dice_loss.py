# -*- coding: utf-8 -*-
# @Time    : 2020-04-22 11:11
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : dice_loss.py

import tensorflow as tf

from aispace.utils.tf_utils import get_shape

__all__ = [
    "dice_loss",
    "cce_dice_loss"
]


def dice_loss(smooth=1, from_logits=False):
    """
    Ref: Li, Xiaoya, et al. "Dice Loss for Data-imbalanced NLP Tasks." arXiv preprint arXiv:1911.02855 (2019).
    :param gamma:
    :return:
    """
    smooth = 1e-17

    def dice_loss_fun(y_true, y_pred):
        true_rank, pred_rank = y_true.shape.ndims, y_pred.shape.ndims
        if true_rank + 1 == pred_rank:
            label_num = get_shape(y_pred)[-1]
            y_true = tf.one_hot(y_true, label_num)
            true_rank = y_true.shape.ndims
        assert true_rank == pred_rank, \
            f"For the tensor y_true, the actual tensor rank {true_rank} (shape = {get_shape(y_true)}) " \
                f"is not equal to the expected tensor rank {pred_rank}"
        y_true = tf.cast(y_true, tf.float32)

        if from_logits:
            y_pred = tf.math.sigmoid(y_pred)

        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

        y_pred_n = 1.0 - p_t
        intersection = tf.reduce_sum(y_pred_n * y_pred * y_true) + smooth
        sum_value = tf.reduce_sum(y_pred_n * y_pred) + tf.reduce_sum(y_true) + smooth
        dce = intersection / sum_value
        return 1.0 - dce

    return dice_loss_fun


def cce_dice_loss(**kwargs):
    """
    Sum of categorical crossentropy and dice losses:

    math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + dice_loss(A, B)

    :param kwargs:
    :return:
    """

    cce_fun = tf.keras.losses.sparse_categorical_crossentropy
    dice_loss_fun = dice_loss(smooth=kwargs.get("smooth", 1e-17), from_logits=kwargs.get("from_logits", False))
    cce_weight = kwargs.get("cce_weight", 1.0)

    def cce_dice_loss_fun(y_true, y_pred):
        cce_loss_v = tf.reduce_mean(cce_fun(y_true, y_pred, from_logits=kwargs.get("from_logits", False)))
        dice_loss_v = dice_loss_fun(y_true, y_pred)
        loss = cce_loss_v * cce_weight + dice_loss_v
        return loss

    return cce_dice_loss_fun