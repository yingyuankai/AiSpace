# -*- coding: utf-8 -*-
# @Time    : 2019-11-01 15:53
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : bert_tokenizer.py
from typing import Optional, List

from aispace.datasets.vocabulary import Vocabulary
from aispace.datasets.tokenizer.tokenizer_base import BaseTokenizer
from aispace.utils.hparams import Hparams
from aispace.utils.str_utils import truncate_seq_pair
from aispace.utils.tokenizer import BasicTokenizer, WordpieceTokenizer

__all__ = ["BertTokenizer"]


@BaseTokenizer.register("bert_tokenizer")
class BertTokenizer(BaseTokenizer):
    """Pre-trained BERT Tokenizer."""

    def __init__(self, hparams: Hparams):
        super(BertTokenizer, self).__init__(hparams)
        self._vocab = Vocabulary(self._hparams.vocab)

        if hparams.do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=hparams.do_lower_case,
                never_split=hparams.non_split_tokens,
                tokenize_chinese_chars=hparams.tokenize_chinese_chars
            )
        self.word_piece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab.token_to_id,
            unk_token=self.vocab.unk_token
        )

    # def __setstate__(self, state):
    #     self._hparams = state[0]
    #     self._vocab = state[1]
    #     self.basic_tokenizer = state[2]
    #     self.word_piece_tokenizer = state[3]
    #
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["_hparams"] = None
    #     state["_vocab"] = None
    #     state["basic_tokenizer"] = None
    #     state['word_piece_tokenizer'] = None

    def tokenize(self, text: str, ret_full=False):
        split_tokens = []
        raw_tokens = []
        align_mapping = []
        if self._hparams.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                    text, never_split=self.vocab.special_tokens()):
                assert token is not None
                raw_tokens.append(token)
                for sub_token in self.word_piece_tokenizer.tokenize(token):
                    align_mapping.append(len(raw_tokens) - 1)
                    split_tokens.append(sub_token)
            if ret_full is True:
                return split_tokens, raw_tokens, align_mapping
        else:
            split_tokens = self.word_piece_tokenizer.tokenize(text)
        return split_tokens

    def detokenizer(self, tokens: List[str]) -> str:
        r"""Maps a sequence of tokens (string) to a single string."""
        link_word = "" if self._hparams.tokenize_chinese_chars else " "
        out_string = f'{link_word}'.join(tokens).replace(f'{link_word}##', '').strip()
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
        cls_id = self.vocab.cls_id
        sep_id = self.vocab.sep_id

        # transforming for text_a
        token_ids_a = self.vocab.transformer(self.tokenize(text_a))
        assert isinstance(token_ids_a, list)

        # transforming for text_a or text_b
        if text_b:
            token_ids_b = self.vocab.transformer(self.tokenize(text_b))
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

    def decode(self, idx, skip_special_tokens=False):
        return self.vocab.transformer(idx, skip_special_tokens)