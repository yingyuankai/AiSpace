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

        # out_string = f'{link_word}'.join(tokens).replace(f'{link_word}##', '').strip()
        out_string = f'{link_word}'.join(tokens).replace(f'{link_word}##', ''.join([' '] * len(f'{link_word}##'))).strip()
        # out_string = ""
        # for i in range(len(tokens)):
        #     if i == 0:
        #         out_string += tokens[0]
        #         continue
        #     if self._is_english(tokens[i]) and self._is_english(tokens[i - 1]):
        #         out_string += ' ' + tokens[i]
        #         out_string = out_string.replace(f' ##', '').strip()
        #     else:
        #         out_string += link_word + tokens[i]
        #         out_string = out_string.replace(f'##', '').strip()

        return out_string

    def encode(self,
               text_a: str,
               text_b: Optional[str] = None,
               max_seq_length: Optional[int] = None,
               return_mask=False, return_offset=False,
               return_cls_index=False):
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
        if isinstance(text_a, str):
            token_a = self.tokenize(text_a)
        elif isinstance(text_a, (tuple, list)):
            token_a = text_a
        token_ids_a = self.vocab.transformer(token_a)
        assert isinstance(token_ids_a, list)

        # transforming for text_a or text_b
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

            # input_ids = [cls_id] + token_ids_a + [sep_id] + token_ids_b + [sep_id]
            # segment_ids = [0] * (len(token_ids_a) + 2) + [1] * (len(token_ids_b) + 1)
        else:
            token_ids_a = token_ids_a[: max_seq_length - 2]
            # input_ids = [cls_id] + token_ids_a + [sep_id]
            # segment_ids = [0] * len(input_ids)

        input_ids = self.build_inputs_with_special_tokens(token_ids_a, token_ids_b)
        segment_ids = self.create_token_type_ids_from_sequences(token_ids_a, token_ids_b)
        special_token_mask = self.get_special_tokens_mask(token_ids_a, token_ids_b)
        a_mask, b_mask = self.get_tokens_mask(token_ids_a, token_ids_b)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the maximum sequence length.
        input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
        segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))
        input_mask = input_mask + [0] * (max_seq_length - len(input_mask))
        special_token_mask = special_token_mask + [0] * (max_seq_length - len(special_token_mask))
        a_mask = a_mask + [0] * (max_seq_length - len(a_mask))
        if b_mask is not None:
            b_mask = b_mask + [0] * (max_seq_length - len(b_mask))

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
            output['offset'] = 1

        if return_cls_index:
            output['cls_index'] = 0

        return output

        # return input_ids, segment_ids, input_mask

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
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

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
        cls_segment_id = [0]

        if token_ids_1 is None:
            return cls_segment_id + len(token_ids_0 + sep) * [0]
        return cls_segment_id + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def get_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        if token_ids_1 is None:
            return [0] + [1] * len(token_ids_0) + [0], None
        return [0] + [1] * len(token_ids_0) + [0] + [0] * (len(token_ids_1) + 1), \
               [0] * (len(token_ids_0) + 2) + [1] * len(token_ids_1) + [0]

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
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def decode(self, idx, skip_special_tokens=False):
        return self.vocab.transformer(idx, skip_special_tokens)

    def _is_english(self, word: str) -> bool:
        """
        Checks whether `word` is a english word.

        Note: this function is not standard and should be considered for BERT
        tokenization only. See the comments for more details.
        :param word:
        :return:
        """
        flag = True
        for c in word:
            if 'a' <= c <= 'z' or 'A' <= c <= 'Z' or c == '#':
                continue
            else:
                flag = False
                break
        return flag