# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 14:04
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : activations.py

import math
import numpy as np
import tensorflow as tf

__all__ = [
    "gelu",
    "gelu_new",
    "swish",
    "ACT2FN"
]


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    :param input:
    :return:
    """
    # cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.sigmoid(x)


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "elu": tf.keras.activations.elu,
    "hard_sigmoid": tf.keras.activations.hard_sigmoid,
    "linear": tf.keras.activations.linear,
    "relu": tf.keras.activations.relu,
    "selu": tf.keras.activations.selu,
    "sigmoid": tf.keras.activations.sigmoid,
    "softmax": tf.keras.activations.softmax,
    "softplus": tf.keras.activations.softplus,
    "softsign": tf.keras.activations.softsign,
    "tanh": tf.keras.activations.tanh
}