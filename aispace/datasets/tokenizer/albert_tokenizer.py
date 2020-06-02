# -*- coding: utf-8 -*-
# @Time    : 2019-11-01 15:53
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_tokenizer.py
from typing import Optional, List
import logging
import six
import unicodedata
import numpy as np

from aispace.datasets.tokenizer.tokenizer_base import BaseTokenizer
from aispace.utils.hparams import Hparams
from aispace.utils.str_utils import truncate_seq_pair

__all__ = ["AlbertTokenizer"]


logger = logging.getLogger(__name__)

SPIECE_UNDERLINE = u'▁'


@BaseTokenizer.register("albert_tokenizer")
class AlbertTokenizer(BaseTokenizer):
    """Pre-trained ALBERT Tokenizer."""

    def __init__(self, hparams: Hparams):
        super(AlbertTokenizer, self).__init__(hparams)

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use AlbertTokenizer: "
                           "https://github.com/google/sentencepiece pip install sentencepiece")

        self.do_lower_case = self._hparams.do_lower_case
        self.remove_space = self._hparams.remove_space
        self.keep_accents = self._hparams.keep_accents
        self.vocab_file = self._hparams.vocab.filename
        self.special_tokens = self._hparams.vocab.special_tokens

        self._vocab = spm.SentencePieceProcessor()
        self._vocab.Load(self.vocab_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_vocab"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning("You need to install SentencePiece to use AlbertTokenizer: "
                           "https://github.com/google/sentencepiece pip install sentencepiece")
        self._vocab = spm.SentencePieceProcessor()
        self._vocab.Load(self.vocab_file)

    def _preprocess_text(self, inputs):
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if six.PY2 and isinstance(outputs, str):
            outputs = outputs.decode('utf-8')

        if not self.keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string.
            return_unicode is used only for py2
        """
        text = self._preprocess_text(text)
        # note(zhiliny): in some systems, sentencepiece only accepts str for py2
        if six.PY2 and isinstance(text, unicode):
            text = text.encode('utf-8')

        if not sample:
            pieces = self._vocab.EncodeAsPieces(text)
        else:
            pieces = self._vocab.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        # note(zhiliny): convert back to unicode for py2
        if six.PY2 and return_unicode:
            ret_pieces = []
            for piece in new_pieces:
                if isinstance(piece, str):
                    piece = piece.decode('utf-8')
                ret_pieces.append(piece)
            new_pieces = ret_pieces

        return new_pieces

    def detokenizer(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def encode(self,
               text_a: str,
               text_b: Optional[str] = None,
               max_seq_length: Optional[int] = None):
        """Adds special tokens to a sequence or sequence pair and computes the
        corresponding segment ids and input mask for BERT specific tasks.
        The sequence will be truncated if its length is larger than
        ``max_seq_length``.

        A BERT sequence has the following format:
        `[cls_token]` X `[sep_token]`

        A BERT sequence pair has the following format:
        `[cls_token]` A `[sep_token]` B `[sep_token]`

        :param text_a: The first input text
        :param text_b: The second input text
        :param max_seq_length: Maximum sequence length
        :return:
            A tuple of `(input_ids, segment_ids, input_mask)`, where

            - ``input_ids``: A list of input token ids with added
              special token ids.
            - ``segment_ids``: A list of segment ids.
            - ``input_mask``: A list of mask ids. The mask has 1 for real
              tokens and 0 for padding tokens. Only real tokens are
              attended to.
        """
        max_seq_length = max_seq_length or self.max_len

        # sepical tokens
        cls_id = self._vocab.PieceToId(self.special_tokens.CLS)
        sep_id = self._vocab.PieceToId(self.special_tokens.SEP)

        # transforming for text_a
        token_ids_a = self.transformer(self.tokenize(text_a))
        assert isinstance(token_ids_a, list)

        # transforming for text_a or text_b
        if text_b:
            token_ids_b = self.transformer(self.tokenize(text_b))
            assert isinstance(token_ids_b, list)
            # Modifies `token_ids_a` and `token_ids_b` in place so that the
            # total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(token_ids_a, token_ids_b, max_seq_length - 3)

            input_ids = [cls_id] + token_ids_a + [sep_id] + token_ids_b + [sep_id]
            segment_ids = [0] * (len(token_ids_a) + 2) + [1] * (len(token_ids_b) + 1)
        else:
            token_ids_a = token_ids_a[: max_seq_length - 2]
            input_ids = [cls_id] + token_ids_a + [sep_id]
            segment_ids = [0] * len(input_ids)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the maximum sequence length.
        input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
        segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))
        input_mask = input_mask + [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return input_ids, segment_ids, input_mask

    def decode(self, idx):
        return self.transformer(idx)

    def transformer(self, inputs):
        if isinstance(inputs, int) or isinstance(inputs, np.int64):
            output = self._convert_id_to_token(inputs)
        elif isinstance(inputs, six.string_types):
            output = self._convert_token_to_id(inputs)
        elif isinstance(inputs, (list, tuple)):
            output = [self.transformer(itm) for itm in inputs if self.transformer(itm) is not None]
        elif input is None:
            logger.error("input is None")
            output = None
        else:
            print(input)
            print(type(input))
            raise NotImplementedError
        return output

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self._vocab.PieceToId(token)

    def _convert_id_to_token(self, index, return_unicode=True):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        token = self._vocab.IdToPiece(index)
        if six.PY2 and return_unicode and isinstance(token, str):
            token = token.decode('utf-8')
        return token
