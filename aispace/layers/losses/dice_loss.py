# -*- coding: utf-8 -*-
# @Time    : 2020-04-22 11:11
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : dice_loss.py

import tensorflow as tf

from aispace.utils.tf_utils import get_shape

__all__ = [
    "BinaryDceLoss",
    "DceLoss",
    "CceDceLoss"
]


class BinaryDceLoss(tf.keras.losses.Loss):
    def __init__(self,
                 smooth=1e-10,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name="binary_dce_loss"):
        super(BinaryDceLoss, self).__init__(name=name, reduction=reduction)
        self.reduction = reduction
        self.smooth = smooth

    def call(self, y_true, y_pred):
        batch_size = get_shape(y_true)[0]
        y_true = tf.reshape(y_true, [batch_size, -1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.reshape(y_pred, [batch_size, -1])

        # p_t = y_pred * y_true + (1. - y_pred) * (1. - y_true)

        num = tf.reduce_sum((1.0 - y_pred) * y_pred * y_true, axis=1) + self.smooth
        den = tf.reduce_sum((1.0 - y_pred) * y_pred + y_true, axis=1) + self.smooth

        loss = 1 - num / den

        return tf.reduce_sum(loss)

    def get_config(self):
        config = {
            "smooth": self.smooth,
            "reduction": self.reduction
        }
        config.update(self.kwargs)
        base_config = super(BinaryDceLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DceLoss(tf.keras.losses.Loss):
    """
    Ref: Li, Xiaoya, et al. "Dice Loss for Data-imbalanced NLP Tasks." arXiv preprint arXiv:1911.02855 (2019).
    """
    def __init__(self, weight=None, ignore_index=None, name="dce_loss", reduction=tf.keras.losses.Reduction.NONE, **kwargs):
        super(DceLoss, self).__init__(name=name, reduction=reduction)
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

        self.binary_dce_loss = BinaryDceLoss(**self.kwargs)

    def call(self, y_true, y_pred):
        """

        :param y_pred: [batch_size, seq_len, label_size]
        :param y_true:
        :return:
        """
        true_rank, pred_rank = y_true.shape.ndims, y_pred.shape.ndims
        if true_rank + 1 == pred_rank:
            label_num = get_shape(y_pred)[-1]
            y_true = tf.one_hot(y_true, label_num)
            true_rank = y_true.shape.ndims
        assert true_rank == pred_rank, \
            f"For the tensor y_true, the actual tensor rank {true_rank} (shape = {get_shape(y_true)}) " \
                f"is not equal to the expected tensor rank {pred_rank}"

        pred_shape = get_shape(y_pred)

        total_loss = 0
        y_pred = tf.math.softmax(y_pred)

        y_pred = tf.unstack(y_pred, axis=-1)
        y_true = tf.unstack(y_true, axis=-1)

        for i in range(pred_shape[-1]):
            if i != self.ignore_index:
                dice_loss = self.binary_dce_loss(y_true[i], y_pred[i])
                if self.weight is not None:
                    assert len(self.weight) == pred_shape[-1], \
                        f'Expect weight shape [{pred_shape[-1]}], get[{len(self.weight)}]'
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / pred_shape[-1]

    def get_config(self):
        config = {
            "weight": self.weight,
            "ignore_index": self.ignore_index,
        }
        config.update(self.kwargs)
        base_config = super(DceLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CceDceLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, cce_weight=1.0, name="cce_dce_loss", reduction=tf.keras.losses.Reduction.NONE, **kwargs):
        super(CceDceLoss, self).__init__(name=name, reduction=reduction)
        self.cce_weight = cce_weight
        self.from_logits = from_logits

        self.dce_loss = DceLoss(**kwargs)

    def call(self, y_true, y_pred):
        cce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        cce_loss *= self.cce_weight

        dce_loss = self.dce_loss(y_true, y_pred)

        loss = cce_loss + dce_loss
        return loss

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
            "cce_weight": self.cce_weight,
        }
        config.update(self.kwargs)
        base_config = super(CceDceLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def dce_tt_loss(y_true, y_pred):
    num = tf.reduce_sum((1 - y_pred) * y_pred * y_true)
    den = tf.reduce_sum((1 - y_pred) * y_pred * y_true + y_true)
    loss = 1 - num / den
    return loss