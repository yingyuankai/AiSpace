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
    "XlnetTokenizer"
]

logger = logging.getLogger(__name__)


SPIECE_UNDERLINE = "▁"


@BaseTokenizer.register("xlnet_tokenizer")
class XlnetTokenizer(BaseTokenizer):
    """Pre-trained xlnet Tokenizer"""

    def __init__(self, hparams: Hparams):
        super(XlnetTokenizer, self).__init__(hparams)
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

    def _preprocess_text(self, inputs):
        outputs = preprocess_text(inputs,
                                  lower=self.do_lower_case,
                                  remove_space=self.remove_space,
                                  keep_accents=self.keep_accents)

        return outputs

    def tokenize(self, input, sample=False):
        """ Tokenize a string. """
        text = self._preprocess_text(input)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def detokenizer(self, tokens: List[str]):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ")
        return out_string

    def encode(self,
               text_a: str,
               text_b: Optional[str] = None,
               max_seq_length: Optional[int] = None,
               return_mask=False, return_offset=False,
               return_cls_index=False):
        """
        encode text or tokens to ids
        :param text_a:
        :param text_b:
        :param max_seq_length:
        :param return_mask:
        :param return_offset:
        :param return_cls_index:
        :return:
        """
        max_seq_length = max_seq_length or self.max_len

        # sepical tokens
        cls_id = self.vocab.cls_id
        sep_id = self.vocab.sep_id
        pad_id = self.vocab.pad_id

        # transforming for text_a
        if isinstance(text_a, str):
            token_a = self.tokenize(text_a)
        elif isinstance(text_a, (tuple, list)):
            token_a = text_a
        token_ids_a = self.vocab.transformer(token_a)
        assert isinstance(token_ids_a, list)

        # transforming for text_a or text_b
        token_ids_b = None
        if text_b:
            if isinstance(text_b, str):
                token_b = self.tokenize(text_b)
            elif isinstance(text_b, (tuple, list)):
                token_b = text_b
            token_ids_b = self.vocab.transformer(token_b)
            assert isinstance(token_ids_b, list)
            # Modifies `token_ids_a` and `token_ids_b` in place so that the
            # total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(token_ids_a, token_ids_b, max_seq_length - 3)
        else:
            token_ids_a = token_ids_a[: max_seq_length - 2]

        input_ids = self.build_inputs_with_special_tokens(token_ids_a, token_ids_b)
        segment_ids = self.create_token_type_ids_from_sequences(token_ids_a, token_ids_b)
        special_token_mask = self.get_special_tokens_mask(token_ids_a, token_ids_b)
        a_mask, b_mask = self.get_tokens_mask(token_ids_a, token_ids_b)

        input_mask = [1] * len(input_ids)
        offset = max_seq_length - len(input_ids)

        input_ids = [pad_id] * (max_seq_length - len(input_ids)) + input_ids
        segment_ids = [0] * (max_seq_length - len(segment_ids)) + segment_ids
        input_mask = [0] * (max_seq_length - len(input_mask)) + input_mask
        special_token_mask = [0] * (max_seq_length - len(special_token_mask)) + special_token_mask
        a_mask = [0] * (max_seq_length - len(a_mask)) + a_mask
        if b_mask is not None:
            b_mask = [0] * (max_seq_length - len(b_mask)) + b_mask

        assert len(input_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(special_token_mask) == max_seq_length
        assert len(a_mask) == max_seq_length

        output = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "input_mask": input_mask,
        }

        if return_mask:
            output['a_mask'] = a_mask
            output['b_mask'] = b_mask

        if return_offset:
            output['a_offset'] = offset
            if text_b:
                output['b_offset'] = offset + len(token_ids_a) + 1

        if return_cls_index:
            output['cls_index'] = len(input_ids) - 1

        return output

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        An XLNet sequence has the following format:

        - single sequence: ``X <sep> <cls>``
        - pair of sequences: ``A <sep> B <sep> <cls>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        sep = [self.vocab.sep_id]
        cls = [self.vocab.cls_id]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        An XLNet sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 2
        | first sequence    | second sequence     | CLS segment ID

        if token_ids_1 is None, only returns the first portion of the mask (0's).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.vocab.sep_id]
        cls_segment_id = [2]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    def get_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        if token_ids_1 is None:
            return [1] * len(token_ids_0) + [0] * 2, None
        return [1] * len(token_ids_0) + [0] * (len(token_ids_1) + 3), \
               [0] * (len(token_ids_0) + 1) + [1] * len(token_ids_1) + [0] * 2

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.vocab.sep_id, self.vocab.cls_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1, 1]
        return ([0] * len(token_ids_0)) + [1, 1]

    def decode(self, idx, skip_special_tokens=False):
        return self.vocab.transformer(idx, skip_special_tokens)