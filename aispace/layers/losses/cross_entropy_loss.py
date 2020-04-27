# -*- coding: utf-8 -*-
# @Time    : 2020-04-27 20:27
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : cross_entropy_loss.py

import tensorflow as tf


class SigmoidCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name="sigmoid_cross_entropy"):
        super(SigmoidCrossEntropy, self).__init__(name=name, reduction=reduction)
        self.reduction = reduction

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        return tf.reduce_mean(cross_ent)