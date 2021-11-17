# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 19:35
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : pretrained_base.py

import logging
from pathlib import Path
from abc import ABCMeta, abstractmethod
import tensorflow as tf

from aispace.utils.hparams import Hparams
from aispace.utils.file_utils import *

logger = logging.getLogger(__name__)


class BasePretrained(BaseModel):
    __metaclass__ = ABCMeta

    def __init__(self, hparams: Hparams, **kwargs):
        super(BasePretrained, self).__init__(hparams, **kwargs)
        self._MODEL2URL = hparams.family
        self._MODEL_NAME = hparams.name
        self.cache_dir = hparams.cache_dir
        # self.pretrained_model_path = self.download_checkpoint(self._MODEL_NAME, self.cache_dir)

    def download_checkpoint(self, pretrained_model_name, cache_dir=None):
        r"""Download the specified pre-trained checkpoint, and return the
        directory in which the checkpoint is cached.

        Args:
            pretrained_model_name (str): Name of the model checkpoint.
            cache_dir (str, optional): Path to the cache directory. If `None`,
                uses the default directory (user's home directory).

        Returns:
            Path to the cache directory.
        """
        if pretrained_model_name in self._MODEL2URL:
            download_path = self._MODEL2URL[pretrained_model_name]
        else:
            raise ValueError(
                "Pre-trained model not found: {}".format(pretrained_model_name))

        if cache_dir is None:
            cache_path = default_download_dir(self._MODEL_NAME)
        else:
            cache_path = Path(cache_dir)
        cache_path = cache_path / pretrained_model_name

        if not cache_path.exists():
            if isinstance(download_path, list):
                for path in download_path:
                    maybe_download(path, str(cache_path))
            else:
                filename = download_path.split('/')[-1]
                maybe_download(download_path, str(cache_path), extract=True)
                folder = None
                for file in cache_path.iterdir():
                    if file.is_dir():
                        folder = file
                assert folder is not None
                (cache_path / filename).unlink()
                for file in folder.iterdir():
                    file.rename(file.parents[1] / file.name)
                folder.rmdir()
            print("Pre-trained {} checkpoint {} cached to {}".format(
                self._MODEL_NAME, pretrained_model_name, cache_path))
        else:
            print("Using cached pre-trained {} checkpoint from {}.".format(
                self._MODEL_NAME, cache_path))

        return str(cache_path)


    @abstractmethod
    def _init_from_checkpoint(self, pretrained_model_name, cache_dir, scope_name, **kwargs):
        r"""Initialize model parameters from weights stored in the pre-trained
        checkpoint.

        Args:
            pretrained_model_name (str): Name of the pre-trained model.
            cache_dir (str): Path to the cache directory.
            scope_name: Variable scope.
            **kwargs: Additional arguments for specific models.
        """
        raise NotImplementedError