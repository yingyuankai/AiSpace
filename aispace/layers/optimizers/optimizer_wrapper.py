# -*- coding: utf-8 -*-
# @Time    : 2020-06-24 20:10
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : optimizer_wrapper.py

import tensorflow as tf
from typeguard import typechecked
from typing import Union


class OptimizerWrapper(tf.keras.optimizers.Optimizer):
    @typechecked
    def __init__(
        self,
        optimizer: Union[tf.keras.optimizers.Optimizer, str],
        name: str = "OptimizerWrapper",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        self._optimizer = optimizer

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name)

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        return train_op

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        return train_op

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices
        )
        return train_op

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer, **config)

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)
