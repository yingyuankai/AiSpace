# -*- coding: utf-8 -*-
# @Time    : 2019-02-22 16:49
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tfrecord_utils.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def to_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value if isinstance(value, list) else [value]))


def to_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(feature):
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
