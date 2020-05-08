# -*- coding: utf-8 -*-
# @Time    : 2020-05-07 17:34
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : focal_loss.py


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
import tensorflow_addons as tfa
from aispace.utils.tf_utils import get_shape

__all__ = [
    "SparseSoftmaxFocalCrossEntropy"
]


class SparseSoftmaxFocalCrossEntropy(tf.keras.losses.Loss):
    """Implements the sparse focal loss function.
    """

    def __init__(
        self,
        from_logits: bool = False,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "sparse_sigmoid_focal_crossentropy",
    ):
        super().__init__(name=name, reduction=reduction)

        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred_shape = get_shape(y_pred)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, y_pred_shape[-1])
        y_true = tf.reshape(y_true, y_pred_shape)

        loss = softmax_focal_crossentropy(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
        )
        return loss

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        base_config = super().get_config()
        return {**base_config, **config}


def softmax_focal_crossentropy(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    from_logits: bool = False,
) -> tf.Tensor:
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.

    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.nn.softmax(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)