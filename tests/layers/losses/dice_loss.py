# -*- coding: utf-8 -*-
# @Time    : 2020-04-22 19:19
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : dice_loss.py

import os, sys
import tensorflow as tf
import unittest

from aispace.layers.losses.dice_loss import dice_loss


class testDiceLoss(unittest.TestCase):
    def test_dice_loss(self):
        y_true = tf.constant([1, 0, 0])
        y_pre = tf.constant([0.1, 0.8, 0.1])
        loss_func = dice_loss()
        loss = loss_func(y_true, y_pre)
        print(loss)