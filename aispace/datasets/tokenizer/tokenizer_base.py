# -*- coding: utf-8 -*-
# @Time    : 2019-10-31 10:58
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tokenizer_base.py

from abc import ABCMeta, abstractmethod
import logging
from typing import Dict, Optional, List

from aispace.utils.hparams import Hparams
from aispace.utils.registry import Registry

__all__ = [
    "BaseTokenizer"
]


SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
CONFIG_FILE = 'config.json'

logger = logging.getLogger(__name__)


class BaseTokenizer(Registry):
    __metaclass__ = ABCMeta

    _MAX_INPUT_SIZE: Dict[str, Optional[int]]
    _VOCAB_FILE_NAMES: Dict[str, str]

    def __init__(self, hparams: Hparams):
        self._hparams = hparams
        self._vocab = None

    @abstractmethod
    def tokenize(self, input):
        """text to tokens

        :param input:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def detokenizer(self, tokens: List[str]) -> str:
        r""" de + tokenizer, antonym of tokenizer.
        Maps a sequence of tokens (string) in a single string.
        The most simple way to do it is :python:`' '.join(tokens)`, but we
        often want to remove sub-word tokenization artifacts at the same time.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self,
               text_a: str,
               text_b: Optional[str] = None,
               max_seq_length: Optional[int] = None):
        """text to idx

        :param text_a:
        :param text_b:
        :param max_seq_length:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, idx):
        """idx to text

        :param idx:
        :return:
        """
        raise NotImplementedError

    @property
    def vocab(self):
        return self._vocab

    @property
    def max_len(self):
        if not self._hparams.has_key("max_len"):
            logger.error("Must specify the max_len in tokenizer")
            return None
        return self._hparams.get("max_len")

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        r"""Clean up a list of simple English tokenization artifacts like
        spaces before punctuations and abbreviated forms.
        """
        out_string = out_string.replace(' .', '.').replace(' ?', '?'). \
            replace(' !', '!').replace(' ,', ',').replace(" ' ", "'"). \
            replace(" n't", "n't").replace(" 'm", "'m"). \
            replace(" do not", " don't").replace(" 's", "'s"). \
            replace(" 've", "'ve").replace(" 're", "'re")
        return out_string