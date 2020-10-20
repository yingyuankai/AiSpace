# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-04 13:38
# @Author  : yingyuankai@aliyun.com
# @File    : misc.py

import random
import numpy as np
import tensorflow as tf


def set_random_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_visible_devices(gpus_idxs=[]):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        if not gpus_idxs:
            gpus = gpus
        else:
            gpus = [gpus[idx] for idx in gpus_idxs]
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPUs,")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def set_xla(enable_xla=False):
    """Config eager context according to flag values using TF 2.0 API."""
    if enable_xla:
        tf.config.optimizer.set_jit(True)
        # Disable PinToHostOptimizer in grappler when enabling XLA because it
        # causes OOM and performance regression.
        tf.config.optimizer.set_experimental_options(
            {'pin_to_host_optimization': False}
        )
