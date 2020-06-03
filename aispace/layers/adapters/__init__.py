# -*- coding: utf-8 -*-
# @Time    : 2019-11-28 15:26
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : __init__.py.py

__all__ = [
    "ADAPTERS"
]

from .model_adapters import *


ADAPTERS = {
    "tf_huggingface_bert_adapter": tf_huggingface_bert_adapter,
    "tf_huggingface_ernie_adapter": tf_huggingface_ernie_adapter,
    "tf_huggingface_xlnet_adapter": tf_huggingface_xlnet_adapter,
    "tf_huggingface_albert_chinese_adapter": tf_huggingface_albert_chinese_adapter,
    "tf_huggingface_albert_chinese_google_adapter": tf_huggingface_albert_chinese_google_adapter,
    "tf_huggingface_electra_adapter": tf_huggingface_electra_adapter
}