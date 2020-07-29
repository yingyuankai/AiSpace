# coding:utf-8
import unicodedata
from typing import Optional, List, Dict

import spacy
from stanfordnlp.server.client import CoreNLPClient
import jieba
import multiprocessing
import re

__all__ = [
    "SpacyTokenizer",
    "JieBaTokenizer",
    "StanfordTokenizer",
    "BasicTokenizer",
    "WordpieceTokenizer"
]


class SpacyTokenizer(object):
    def __init__(self, fine_grained=False):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'entity'])
        self.fine_grained = fine_grained

    def word_tokenizer(self, doc):
        if not self.fine_grained:
            doc = self.nlp(doc)
            tokens = [token.text for token in doc]
            token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
            return tokens, token_spans
        sentence = doc
        tokens = []
        token_spans = []
        cur = 0
        pattern = u'-|–|—|:|’|\.|,|\[|\?|\(|\)|~|\$|/'
        for next in re.finditer(pattern, sentence):
            for token in self.nlp(sentence[cur:next.regs[0][0]]):
                if token.text.strip() != '':
                    tokens.append(token.text)
                    token_spans.append((cur + token.idx, cur + token.idx + len(token.text)))
            tokens.append(sentence[next.regs[0][0]:next.regs[0][1]])
            token_spans.append((next.regs[0][0], next.regs[0][1]))
            cur = next.regs[0][1]
        for token in self.nlp(sentence[cur:]):
            if token.text.strip() != '':
                tokens.append(token.text)
                token_spans.append((cur + token.idx, cur + token.idx + len(token.text)))
        return tokens, token_spans

    def word_tokenizer_parallel(self, docs):
        docs = [doc for doc in self.nlp.pipe(docs, batch_size=64, n_threads=multiprocessing.cpu_count())]
        tokens = [[token.text for token in doc] for doc in docs]
        token_spans = [[(token.idx, token.idx + len(token.text)) for token in doc] for doc in docs]
        return tokens, token_spans


class JieBaTokenizer(object):
    """
    only for chinese tokenize,no pos/ner feature function
    """

    def __init__(self):
        self.tokenizer = jieba

    def word_tokenizer(self, doc):
        tokens = self.tokenizer.cut(doc)
        tokens = '<split>'.join(tokens).split('<split>')
        start = 0
        token_spans = []
        for token in tokens:
            token_spans.append((start, start + len(token)))
            start += len(token)
        return tokens, token_spans


class StanfordTokenizer(object):
    def __init__(self, language='zh', annotators='ssplit tokenize', timeout=30000, memory="4G"):
        if language == 'zh':
            CHINESE_PROPERTIES = {
                "tokenize.language": "zh",
                "segment.model": "edu/stanford/nlp/models/segmenter/chinese/ctb.gz",
                "segment.sighanCorporaDict": "edu/stanford/nlp/models/segmenter/chinese",
                "segment.serDictionary": "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz",
                "segment.sighanPostProcessing": "true",
                "ssplit.boundaryTokenRegex": "[.。]|[!?！？]+",
            }
        else:
            CHINESE_PROPERTIES = {}
        self.client = CoreNLPClient(annotators=annotators, timeout=timeout, memory=memory,
                                    properties=CHINESE_PROPERTIES)

    def word_tokenizer(self, doc):
        try:
            annotated = self.client.annotate(doc)
            tokens, token_spans = [], []
            for sentence in annotated.sentence:
                for token in sentence.token:
                    tokens.append(token.word)
                    token_spans.append((token.beginChar, token.endChar))
            return tokens, token_spans
        except Exception as e:
            return None, None


class BasicTokenizer:
    r"""Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case: Whether to lower case the input.
        never_split: A list of tokens not to split.
        tokenize_chinese_chars: Whether to tokenize Chinese characters.
            This should likely be deactivated for Japanese:
            see:
            `https://github.com/huggingface/pytorch-pretrained-BERT/issues/328`
    """

    def __init__(self, do_lower_case: bool = True,
                 never_split: Optional[List[str]] = None,
                 tokenize_chinese_chars: bool = True):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text: str,
                 never_split: Optional[List[str]] = None) -> \
            List[str]:
        r"""Basic tokenization of a piece of text.

        Split on white spaces only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            text: An input string.
            never_split: A list of tokens not to split.
        """
        never_split = self.never_split + (never_split
                                          if never_split is not None else [])
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # see: https://github.com/google-research/bert/blob/master/
        # tokenization.py#L201

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    @classmethod
    def _run_strip_accents(cls, text: str) -> str:
        r"""Strips accents from a piece of text.

        Example:
            accented_string = 'Málaga'
            _run_strip_accents(accented_string)  # 'Malaga'
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @classmethod
    def _run_split_on_punc(cls, text: str,
                           never_split: Optional[List[str]] = None) -> \
            List[str]:
        r"""Splits punctuation on a piece of text.

        Example:
            text = 'Texar-PyTorch is an open-source toolkit based on PyTorch.'
            _run_split_on_punc(text)
            # ['Texar', '-', 'PyTorch is an open', '-',
            # 'source toolkit based on PyTorch', '.']
        """
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        r"""Adds whitespace around any CJK character.

        Example:
            text = '今天天气不错'
            _tokenize_chinese_chars(text)
            # ' 今  天  天  气  不  错 '
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp) or _is_punctuation(char):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @classmethod
    def _is_chinese_char(cls, cp: int) -> bool:
        r"""Checks whether cp is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode
        # block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean
        # characters, despite its name. The modern Korean Hangul alphabet is a
        # different block, as is Japanese Hiragana and Katakana. Those
        # alphabets are used to write space-separated words, so they are not
        # treated specially and handled like the all of the other languages.
        if ((0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False

    @classmethod
    def _clean_text(cls, text: str) -> str:
        r"""Performs invalid character removal and whitespace cleanup on text.

        Example:
            text = 'Texar-PyTorch\tis an open-source\ntoolkit based on PyTorch.'
            _clean_text(text)
            # 'Texar-PyTorch is an open-source toolkit based on PyTorch.'
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer:
    r"""Runs WordPiece tokenization."""

    def __init__(self, vocab: Dict[str, int],
                 unk_token: str,
                 max_input_chars_per_word: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        r"""Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Args:
            text: A single token or whitespace separated tokens. This should
                have already been passed through `BasicTokenizer`.

        Returns:
            A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            assert token is not None
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def whitespace_tokenize(text: str) -> List[str]:
    r"""Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens: List[str] = text.split()
    return tokens


def _is_whitespace(char: str) -> bool:
    r"""Checks whether `char` is a whitespace character.

    Note: this function is not standard and should be considered for BERT
    tokenization only. See the comments for more details.
    """
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in (" ", "\t", "\n", "\r"):
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char: str) -> bool:
    r"""Checks whether `char` is a control character.

    Note: this function is not standard and should be considered for BERT
    tokenization only. See the comments for more details.
    """
    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char: str) -> bool:
    r"""Checks whether `char` is a punctuation character.

    Note: this function is not standard and should be considered for BERT
    tokenization only. See the comments for more details.
    """
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False