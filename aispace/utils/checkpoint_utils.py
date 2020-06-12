# -*- coding: utf-8 -*-
# @Time    : 2019-12-18 14:20
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : checkpoint_utils.py

__all__ = [
    "average_checkpoints"
]

import os

import tensorflow as tf


def average_checkpoints(model, prefix_or_checkpints, num_last_checkpoints=None, ckpt_weights=None):
    """average checkpoints

    :param model_variables:
    :param checkpints:
    :param num_last_checkpoints:
    :return:
    """
    avg_weights = None

    if isinstance(prefix_or_checkpints, (list, tuple)):
        ckpts = prefix_or_checkpints
    elif prefix_or_checkpints.find(',') != -1 or not os.path.exists(prefix_or_checkpints):
        # checkpoints
        ckpts = prefix_or_checkpints.split(",")
    elif os.path.isdir(prefix_or_checkpints):
        # prefix, i.e., directory of checkpoints
        ckpts = tf.train.get_checkpoint_state(prefix_or_checkpints).all_model_checkpoint_paths
        if num_last_checkpoints:
            ckpts = ckpts[-num_last_checkpoints:]
            ckpt_weights = ckpt_weights[-num_last_checkpoints:]
    else:
        raise ValueError(f"{prefix_or_checkpints} is wrong!")

    if ckpt_weights is None:
        ckpt_weights = [1.] * len(ckpts)

    assert len(ckpt_weights) == len(ckpts), \
        ValueError(f"size of ckpt_weights ({len(ckpt_weights)}) must be equal to the size of ckpts ({len(ckpts)}).")

    for idx, ckpt in enumerate(ckpts):
        model.load_weights(ckpt)
        model_weights = model.get_weights()
        model_weights = _weights_time(model_weights, ckpt_weights[idx])
        if idx == 0:
            avg_weights = model_weights
            continue
        avg_weights = _weights_add(avg_weights, model_weights)

    avg_weights = _weights_div(avg_weights, sum(ckpt_weights))
    model.set_weights(avg_weights)


def _weights_add(weights1, weights2):
    weights = [w1 + w2 for w1, w2 in zip(weights1, weights2)]
    return weights


def _weights_div(weights, num):
    weights = [w / num for w in weights]
    return weights


def _weights_time(weights, ckpt_w):
    weights = [w * ckpt_w for w in weights]
    return weights