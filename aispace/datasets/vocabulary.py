# -*- coding: utf-8 -*-
# @Time    : 2019-10-31 10:46
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : vocabulary.py
from typing import List, Optional
import six
import numpy as np
import logging
import tensorflow as tf
from collections import defaultdict

from aispace.utils.hparams import Hparams

__all__ = [
    "Vocabulary"
]

logger = logging.getLogger(__name__)


class Vocabulary(object):
    SPECIAL_TOKENS = ["PAD", "UNK", "BOS", "EOS", "SEP", "CLS", "MASK"]

    def __init__(self, hparams: Hparams):
        self._hparams = hparams
        self._special_tokens: Hparams = self._hparams.special_tokens
        self._filename = self._hparams.filename

        # self._token_to_id_table, \
        # self._id_to_token_table, \
        self._token_to_id_dict, \
        self._id_to_token_dict = self.load(self._filename)

    def load(self, filename):
        from aispace.utils.io_utils import load_vocab
        vocab = load_vocab(filename)

        # valid special tokens
        special_token_in_vocab = []
        for st in self.SPECIAL_TOKENS:
            if not hasattr(self._special_tokens, st):
                continue
            special_token_word = getattr(self._special_tokens, st)
            if special_token_word in vocab:
                logger.warning(f"{st} already exists in the vocabulary: {special_token_word}")
                special_token_in_vocab.append(st)

        # merge vocab with special tokens
        vocab = [self._special_tokens.get(st) for st in self.SPECIAL_TOKENS
                 if st in self.special_tokens() and st not in special_token_in_vocab] + vocab
        unk_id = [idx for idx, word in enumerate(vocab) if word == self.unk_token][0]
        vocab_size = len(vocab)
        vocab_idx = np.arange(vocab_size)
        # build dict
        id_to_token_dict = self.make_dict(vocab_idx, vocab, self.unk_token)
        token_to_id_dict = self.make_dict(vocab, vocab_idx, unk_id)

        # build hash table
        # id_to_token_table = self._make_tensor_table(vocab_idx, vocab, tf.int64, tf.string, self.unk_token)
        # token_to_id_table = self._make_tensor_table(vocab, vocab_idx, tf.string, tf.int64, unk_id)

        # return token_to_id_table, id_to_token_table, token_to_id_dict, id_to_token_dict
        return token_to_id_dict, id_to_token_dict

    # def _make_tensor_table(self, keys, values, key_type, value_type, default_value):
    #     table = tf.lookup.StaticHashTable(
    #         tf.lookup.KeyValueTensorInitializer(
    #             keys, values, key_dtype=key_type, value_dtype=value_type
    #         ), default_value
    #     )
    #     return table
    def make_dict(self, keys, values, default_value):
        dict_ = defaultdict(lambda: default_value)
        for k, v in zip(keys, values):
            dict_[k] = v
        return dict_

    def transformer(self, input, skip_special_tokens: Optional[bool] = False):
        if isinstance(input, int) or isinstance(input, np.int64):
            if skip_special_tokens and input in self.special_token_ids():
                output = None
            else:
                output = self._id_to_token_dict.get(input, self._special_tokens.UNK)
        elif isinstance(input, six.string_types):
            output = self._token_to_id_dict.get(input, self._token_to_id_dict[self._special_tokens.UNK])
        elif isinstance(input, (list, tuple)):
            output = []
            for item in input:
                tmp = self.transformer(item, skip_special_tokens)
                if tmp is None:
                    continue
                output.append(tmp)
        # elif tf.is_tensor(input):
        #     input_type = input.dtype
        #     if input_type is tf.int64 or input_type is tf.int32 or input_type is tf.int16 or input_type is tf.int8:
        #         output = self._id_to_token_table.lookup(tf.cast(input, tf.int64))
        #     elif input_type is tf.string:
        #         output = self._token_to_id_table.lookup(input)
        #     else:
        #         raise NotImplementedError
        elif input is None:
            logger.error("input is None")
            output = None
        else:
            print(input)
            print(type(input))
            raise NotImplementedError
        return output

    @property
    def token_to_id(self):
        return self._token_to_id_dict

    @property
    def id_to_token(self):
        return self._id_to_token_dict

    @property
    def pad_token(self):
        if "PAD" not in self._special_tokens:
            logger.error("Using pad_token, but it is not set yet.")
        return self._special_tokens.PAD

    @property
    def pad_id(self):
        return self.transformer(self._special_tokens.PAD)

    @property
    def unk_token(self):
        if "UNK" not in self._special_tokens:
            logger.error("Using unk_token, but it is not set yet.")
        return self._special_tokens.UNK

    @property
    def unk_id(self):
        return self.transformer(self._special_tokens.UNK)

    @property
    def bos_token(self):
        if "BOS" not in self._special_tokens:
            logger.error("Using bos_token, but it is not set yet.")
        return self._special_tokens.BOS

    @property
    def bos_id(self):
        return self.transformer(self._special_tokens.BOS)

    @property
    def eos_token(self):
        if "EOS" not in self._special_tokens:
            logger.error("Using eos_token, but it is not set yet.")
        return self._special_tokens.EOS

    @property
    def eos_id(self):
        return self.transformer(self._special_tokens.EOS)

    @property
    def sep_token(self):
        if "SEP" not in self._special_tokens:
            logger.error("Using sep_token, but it is not set yet.")
        return self._special_tokens.SEP

    @property
    def sep_id(self):
        return self.transformer(self._special_tokens.SEP)

    @property
    def cls_token(self):
        if "CLS" not in self._special_tokens:
            logger.error("Using cls_token, but it is not set yet.")
        return self._special_tokens.CLS

    @property
    def cls_id(self):
        return self.transformer(self._special_tokens.CLS)

    @property
    def mask_token(self):
        if "MASK" not in self._special_tokens:
            logger.error("Using mask_token, but it is not set yet.")
        return self._special_tokens.MASK

    @property
    def mask_id(self):
        return self.transformer(self._special_tokens.MASK)

    def special_tokens(self) -> list:
        return list(self._special_tokens.values())

    def special_token_ids(self) -> List[int]:
        return [self.transformer(token) for token in self.special_tokens()]

    @pad_token.setter
    def pad_token(self, value):
        self._special_tokens.PAD = value

    @unk_token.setter
    def unk_token(self, value):
        self._special_tokens.UNK = value

    @bos_token.setter
    def bos_token(self, value):
        self._special_tokens.BOS = value

    @eos_token.setter
    def eos_token(self, value):
        self._special_tokens.EOS = value

    @sep_token.setter
    def sep_token(self, value):
        self._special_tokens.SEP = value

    @cls_token.setter
    def cls_token(self, value):
        self._special_tokens.CLS = value

    @mask_token.setter
    def mask_token(self, value):
        self._special_tokens.MASK = value

    def vocab_size(self):
        return len(self._token_to_id_dict)

    def __len__(self):
        return self.vocab_size() + len(self.special_tokens())
