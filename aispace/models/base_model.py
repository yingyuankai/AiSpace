# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-05 10:27
# @Author  : yingyuankai@aliyun.com
# @File    : base_model.py

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.registry import Registry


__all__ = [
    "BaseModel"
]


class BaseModel(tf.keras.Model, Registry):
    r""" Base class for all configuration classes.
            Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.
            Note:
                A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
                It only affects the model's configuration.
            Class attributes (overridden by derived classes):
                - ``pretrained_config_archive_map``: a python ``dict`` with `shortcut names` (string) as keys and `url` (string) of associated pretrained model configurations as values.
                - ``model_type``: a string that identifies the model type, that we serialize into the JSON file, and that we use to recreate the correct object in :class:`~transformers.AutoConfig`.
            Args:
                finetuning_task (:obj:`string` or :obj:`None`, `optional`, defaults to :obj:`None`):
                    Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
                num_labels (:obj:`int`, `optional`, defaults to `2`):
                    Number of classes to use when the model is a classification model (sequences/tokens)
                output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
                    Should the model returns attentions weights.
                output_hidden_states (:obj:`string`, `optional`, defaults to :obj:`False`):
                    Should the model returns all hidden-states.
                torchscript (:obj:`bool`, `optional`, defaults to :obj:`False`):
                    Is the model used with Torchscript (for PyTorch models).
        """
    __metaclass__ = ABCMeta

    def __init__(self, hparams: Hparams, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self._hparams = hparams

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @abstractmethod
    def deploy(self):
        raise NotImplementedError