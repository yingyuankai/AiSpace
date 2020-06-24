# -*- coding: utf-8 -*-
# @Time    : 2020-06-24 14:52
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : lr_multiplier.py


import tensorflow as tf

from aispace.layers.optimizers.optimizer_wrapper import OptimizerWrapper

__all__ = [
    "LRMultiplier"
]


class LRMultiplier(OptimizerWrapper):
    """Learning rate multiplier wrapper for optimizers.
       ref: "https://pypi.org/project/keras-lr-multiplier/"
    """
    def __init__(self, optimizer, multipliers, name='lr_multiplier', **kwargs):
        super().__init__(optimizer, name, **kwargs)

        self._optimizer = optimizer
        self.multipliers = multipliers

        if hasattr(self._optimizer, 'learning_rate'):
            self.lr_attr = 'learning_rate'
        else:
            self.lr_attr = 'lr'

    def _get_multiplier(self, name):
        multiplier, prefix_len = 1.0, 0
        if self.multipliers is None:
            return 1.0
        for key, val in self.multipliers.items():
            if name.startswith(key):
                if len(key) > prefix_len:
                    prefix_len = len(key)
                    multiplier = val
        return multiplier

    def get_updates(self, loss, params):
        if len(self.updates) > 0:
            return self.updates
        multiplies = {}
        for param in params:
            multiplier = self._get_multiplier(param.name)
            if multiplier not in multiplies:
                multiplies[multiplier] = []
            multiplies[multiplier].append(param)

        self.updates, self.weights = [], []
        origin_lr = getattr(self, self.lr_attr)
        for i, (multiplier, params) in enumerate(multiplies.items()):
            lr = origin_lr
            if callable(multiplier):
                lr = lr * multiplier(tf.cast(self._optimizer.iterations, tf.keras.backend.floatx()))
            elif multiplier != 1.0:
                lr = lr * multiplier
            setattr(self, self.lr_attr, lr)
            with tf.keras.backend.name_scope('Group_{}'.format(i)):
                self.updates += self._optimizer.get_updates(loss, params)
            for w in self._optimizer.weights:
                if w not in self.weights:
                    self.weights.append(w)
        setattr(self, self.lr_attr, origin_lr)

        return self.updates

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            'multipliers': self.multipliers
        }
        base_config = super().get_config()
        return {**base_config, **config}

