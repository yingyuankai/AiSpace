# -*- coding: utf-8 -*-
# @Time    : 2020-06-01 20:36
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : xlnet_tokenizer.py

import logging
import unicodedata
from typing import List, Optional

from aispace.datasets.vocabulary import Vocabulary
from aispace.datasets.tokenizer.tokenizer_base import BaseTokenizer
from aispace.utils.hparams import Hparams
from aispace.utils.str_utils import truncate_seq_pair, preprocess_text


__all__ = [
    "CPMTokenizer"
]

logger = logging.getLogger(__name__)

try:
    import jieba
except ImportError:
    logger.warning(
        "You need to install jieba to use CPMTokenizer."
        "pip install jieba"
    )
    raise

SPIECE_UNDERLINE = "▁"


@BaseTokenizer.register("cpm_tokenizer")
class CPMTokenizer(BaseTokenizer):
    """Pre-trained xlnet Tokenizer"""

    def __init__(self, hparams: Hparams):
        super(CPMTokenizer, self).__init__(hparams)
        self._vocab = Vocabulary(self._hparams.vocab)

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise



        self.do_lower_case = self._hparams.do_lower_case
        self.remove_space = self._hparams.remove_space
        self.keep_accents = self._hparams.keep_accents

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self._hparams.vocab.filename)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def tokenize(self, input, sample=False):
        """ Tokenize a string. """
        bpe_tokens = []
        seg_list = [x.translate(self.translator) for x in jieba.cut(input, cut_all=False)]
        new_seg = " ".join(seg_list)
        tmp_bpe_tokens = self.sp.encode(new_seg, out_type=str)
        bpe_tokens.extend(tmp_bpe_tokens)
        return bpe_tokens

    def detokenizer(self, tokens: List[str]):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ")
        return out_string

    def encode(self,
               text_a: str,
               text_b=None,
               max_seq_length: Optional[int] = None):
        """
        encode text or tokens to ids
        :param text_a:
        :param text_b:
        :param max_seq_length:
        :return:
        """
        max_seq_length = max_seq_length or self.max_len

        # transforming for text_a
        if isinstance(text_a, str):
            token_a = self.tokenize(text_a)
        elif isinstance(text_a, (tuple, list)):
            token_a = text_a
        token_ids_a = self.vocab.transformer(token_a)
        assert isinstance(token_ids_a, list)

        if len(token_ids_a) > max_seq_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(len(token_ids_a), max_seq_length)
            )
            token_ids_a = token_ids_a[: max_seq_length]

        output = {
            "input_ids": token_ids_a,
        }

        return output

    def decode(self, idx, skip_special_tokens=False):
        text = self.sp.decode([self.decoder[x] for x in idx])
        text = text.replace('\u2582', ' ').replace('\u2583', '\n').replace('\u2584', '<eod>')
        return text