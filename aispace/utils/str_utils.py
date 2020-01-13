# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 11:08
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : str_utils.py
from typing import Union, List


def truncate_seq_pair(tokens_a: Union[List[int], List[str]],
                      tokens_b: Union[List[int], List[str]],
                      max_length: int):
    r"""Truncates a sequence pair in place to the maximum length.

    This is a simple heuristic which will always truncate the longer sequence
    one token at a time. This makes more sense than truncating an equal
    percent of tokens from each, since if one sequence is very short then
    each token that's truncated likely contains more information than a
    longer sequence.

    Example:

    .. code-block:: python

        tokens_a = [1, 2, 3, 4, 5]
        tokens_b = [6, 7]
        truncate_seq_pair(tokens_a, tokens_b, 5)
        tokens_a  # [1, 2, 3]
        tokens_b  # [6, 7]

    Args:
        tokens_a: A list of tokens or token ids.
        tokens_b: A list of tokens or token ids.
        max_length: maximum sequence length.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()