# -*- coding: utf-8 -*-
# @Time    : 2020-05-20 12:40
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : f1_score.py

import tensorflow as tf
import tensorflow_addons as tfa

from aispace.utils.tf_utils import get_shape

__all__ = [
    "SparseF1Score"
]


class SparseF1Score(tfa.metrics.FBetaScore):
    def __init__(
        self,
        num_classes,
        average: str = None,
        threshold=None,
        name: str = "sparse_f1_score",
        dtype=None,
        **kwargs
    ):
        super(SparseF1Score, self).__init__(num_classes, average, 1.0, name=name, dtype=dtype)
        self.threshold = threshold

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, self.num_classes)
        y_true = tf.reshape(y_true, [-1, self.num_classes])
        y_pred = tf.reshape(y_pred, [-1, self.num_classes])

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)

        self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
        self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
        self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))
        self.weights_intermediate.assign_add(_count_non_zero(y_true))