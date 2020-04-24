# -*- coding: utf-8 -*-
# @Time    : 2020-04-22 19:19
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : dice_loss.py

import os, sys
import tensorflow as tf
import unittest

from aispace.layers.losses.dice_loss import DceLoss, CceDceLoss


def dce_tt_loss(y_true, y_pred):
    smooth = 1e-10
    num = (1. - y_pred) * y_pred * y_true + smooth
    den = (1. - y_pred) * y_pred * y_true + y_true + smooth
    loss = 1. - num / den
    loss = tf.reduce_mean(loss)
    return loss


class testDceLoss(unittest.TestCase):
    def test_dce_loss(self):
        y_true = tf.constant([[0., 1., 0.]])
        y_pre = tf.constant([[0.0, 1., 0.0]])

        dce_loss = DceLoss(smooth=1e-10)
        loss = dce_loss(y_true, y_pre)
        # loss = dce_tt_loss(y_true, y_pre)
        print(loss)

    def test_cce_dce_loss(self):
        y_true = tf.constant([[1]])
        y_pre = tf.constant([[[0.1, 1.0, 0.1]]])

        cce_dce_loss = CceDceLoss(from_logits=True, label_num=3, seq_len=1)
        loss = cce_dce_loss(y_true, y_pre)
        print(loss)

    def test_focal_loss(self):
        from tensorflow_addons.losses import sigmoid_focal_crossentropy
        y_true = tf.constant([1, 0, 0])
        y_pre = tf.constant([0.1, 0.8, 0.1])
        loss = sigmoid_focal_crossentropy(y_true=y_true, y_pred=y_pre)
        print(loss)
        y_pre = tf.constant([0.7, 0.1, 0.1])
        loss = sigmoid_focal_crossentropy(y_true=y_true, y_pred=y_pre)
        print(loss)

