# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 11:08
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : str_utils.py

from typing import Union, List
import unicodedata
import six


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


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
