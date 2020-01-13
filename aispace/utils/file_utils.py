# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 19:53
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : file_utils.py

import sys, os
from pathlib import Path
import tensorflow as tf
import tarfile
import zipfile
from six.moves import urllib
import requests
import logging

__all__ = [
    "default_download_dir",
    "set_default_download_dir",
    "DEFAULT_AISPACE_DOWNLOAD_DIR",
    "maybe_create_dir",
]

DEFAULT_AISPACE_DOWNLOAD_DIR = None

logger = logging.getLogger(__name__)


def default_download_dir(name):
    r"""Return the directory to which packages will be downloaded by default.
    """
    global DEFAULT_AISPACE_DOWNLOAD_DIR  # pylint: disable=global-statement
    if DEFAULT_AISPACE_DOWNLOAD_DIR is None:
        if sys.platform == 'win32' and 'AISPACEDATA' in os.environ:
            # On Windows, use %APPDATA%
            home_dir = Path(os.environ['AISPACEDATA'])
        else:
            # Otherwise, install in the user's home directory.
            home_dir = Path(os.environ["HOME"])

        if os.access(str(home_dir), os.W_OK):
            DEFAULT_AISPACE_DOWNLOAD_DIR = home_dir / 'aispace_data'
        else:
            raise ValueError("The path {} is not writable. Please manually "
                             "specify the download directory".format(home_dir))

    if not DEFAULT_AISPACE_DOWNLOAD_DIR.exists():
        DEFAULT_AISPACE_DOWNLOAD_DIR.mkdir(parents=True)

    return DEFAULT_AISPACE_DOWNLOAD_DIR / name


def set_default_download_dir(path):
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValueError("`path` must be a string or a pathlib.Path object")

    if not os.access(str(path), os.W_OK):
        raise ValueError(
            "The specified download directory {} is not writable".format(path))

    global DEFAULT_AISPACE_DOWNLOAD_DIR  # pylint: disable=global-statement
    DEFAULT_AISPACE_DOWNLOAD_DIR = path


def maybe_create_dir(dirname):
    """Creates directory if doesn't exist
    """
    if not tf.io.gfile.isdir(dirname):
        tf.io.gfile.makedirs(dirname)
        return True
    return False
