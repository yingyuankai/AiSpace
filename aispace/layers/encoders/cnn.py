# -*- coding: utf-8 -*-
# @Time    : 2020-05-07 13:57
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : cnn.py

import tensorflow as tf

from aispace.utils.tf_utils import get_bias_initializer, get_initializer

from aispace.layers.activations import ACT2FN

__all__ = [
    "Dgcnn",
    "DgcnnBlock",
    "Textcnn",
    "TextcnnBlock"
]


class DgcnnBlock(tf.keras.layers.Layer):
    """
    ref: https://kexue.fm/archives/5409
    """
    def __init__(self, filters, windows, dilations, stddev=0.1, **kwargs):
        super(DgcnnBlock, self).__init__(**kwargs)

        assert isinstance(windows, (tuple, list)), \
            ValueError("windows must be a list like value, like [3, 3, 3,...]")
        assert isinstance(dilations, (tuple, list)), \
            ValueError("dilations must be a list like value, like [1, 2, 4,...]")
        assert len(windows) == len(dilations), \
            ValueError("The length of windows and dilations must be same.")

        self.convs = [Dgcnn(filters, windows[i], dilations[i], stddev, name=f"conv1d_{i}") for i in range(len(dilations))]

    def call(self, input, **kwargs):
        training = kwargs.get('training', False)
        output = input

        for i, conv in enumerate(self.convs):
            output = self.convs[i](output, training=training)

        return output


class Dgcnn(tf.keras.layers.Layer):
    """
    X = X + Conv1D_1(X) * Sigmoid(Conv1D_2(X) + noise)
    ref: 1. "Convolutional Sequence to Sequence Learning"
         2. https://kexue.fm/archives/5409
    """
    def __init__(self, filters, window=3, dilation=1, stddev=0.1, initializer_range=0.02, **kwargs):
        super(Dgcnn, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv1D(
            filters,
            window,
            dilation_rate=dilation,
            padding='SAME',
            kernel_initializer=get_initializer(initializer_range),
            bias_initializer=get_bias_initializer('conv'))

        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            window,
            dilation_rate=dilation,
            padding='SAME',
            kernel_initializer=get_initializer(initializer_range),
            bias_initializer=get_bias_initializer('conv'))

        self.noise = tf.keras.layers.GaussianNoise(stddev)

    def call(self, input, **kwargs):
        training = kwargs.get('training', False)
        output = self.conv1(input)
        gate = tf.keras.activations.sigmoid(self.noise(self.conv2(input), training=training))
        output = input + output * gate
        return output


class Textcnn(tf.keras.layers.Layer):
    def __init__(self, filter, window, initializer_range=0.02, **kwargs):
        super(Textcnn, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filter,
            window,
            padding='SAME',
            kernel_initializer=get_initializer(initializer_range),
            bias_initializer=get_bias_initializer('conv')
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.act_fn = ACT2FN['relu']
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()

    def call(self, input, **kwargs):
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.act_fn(output)
        output = self.max_pool(output)
        return output


class TextcnnBlock(tf.keras.layers.Layer):
    def __init__(self, filters, windows, initializer_range=0.02, **kwargs):
        super(TextcnnBlock, self).__init__(**kwargs)
        self.convs = [Textcnn(filters[i], windows[i], initializer_range) for i in range(len(filters))]

    def call(self, input, **kwargs):
        outputs = [conv(input) for conv in self.convs]
        output = tf.concat(outputs, axis=-1)
        return output