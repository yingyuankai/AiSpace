# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 15:38
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : tnew_transformer.py


import os
import logging
from tqdm import tqdm
import json
import numpy as np
from .base_transformer import BaseTransformer
from aispace.datasets import BaseTokenizer
from aispace.utils.io_utils import json_dumps
from aispace.utils.str_utils import preprocess_text

__all__ = [
    "TnewsTransformer",
    "CMRC2018Transformer"
]

logger = logging.getLogger(__name__)


@BaseTransformer.register("glue_zh/tnews")
class TnewsTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(TnewsTransformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

        # json dir
        self.json_dir = os.path.join(kwargs.get("data_dir", self._hparams.dataset.data_dir), "json")

    def transform(self, data_path, split="train"):
        # output_path_base = os.path.join(os.path.dirname(data_path), "json")
        # if not os.path.exists(output_path_base):
        #     os.makedirs(output_path_base)
        # output_path = os.path.join(output_path_base, f"{split}.json")
        with open(data_path, "r", encoding="utf8") as inf:
            # with open(output_path, "w", encoding="utf8") as ouf:
            for line in inf:
                if not line: continue
                line = line.strip()
                if len(line) == 0: continue
                line_json = json.loads(line)
                sentence = line_json.get("sentence", "").strip()
                keywords = line_json.get("keywords", "").strip()
                if len(sentence) == 0 and len(keywords) == 0: continue
                encode_info = self.tokenizer.encode(sentence, keywords)
                input_ids, token_type_ids, attention_mask = \
                    encode_info['input_ids'], encode_info['segment_ids'], encode_info['input_mask']
                label = line_json.get("label_desc", "news_story")
                item = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                    "label": label
                }
                yield item
                # new_line = f"{json_dumps(item)}\n"
                # ouf.write(new_line)
        # return output_path


@BaseTransformer.register("glue_zh/cmrc2018")
class CMRC2018Transformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super(CMRC2018Transformer, self).__init__(hparams, **kwargs)

        # tokenizer
        self.tokenizer = \
            BaseTokenizer. \
                by_name(self._hparams.dataset.tokenizer.name) \
                (self._hparams.dataset.tokenizer)

        # json dir
        self.json_dir = os.path.join(kwargs.get("data_dir", self._hparams.dataset.data_dir), "json")

        self.max_query_length = self._hparams.dataset.tokenizer.max_query_length
        self.doc_stride = self._hparams.dataset.tokenizer.doc_stride

    def transform(self, data_path, split="train"):
        """
        ref: https://github.com/stevezheng23/xlnet_extension_tf/blob/master/run_squad.py
        :param data_path:
        :param split:
        :return:
        """
        # output_path_base = os.path.join(os.path.dirname(data_path), "json")
        # if not os.path.exists(output_path_base):
        #     os.makedirs(output_path_base)
        # output_path = os.path.join(output_path_base, f"{split}.json")

        unique_id = 10000000

        # with open(output_path, "w", encoding="utf8") as ouf:
        for e_i, example in enumerate(self._read_next(data_path)):
            # if e_i >= 100:
            #     break
            # if example['qas_id'] != "TRAIN_1756_QUERY_3":
            #     continue
            question_text = example['question_text']
            if self._hparams.dataset.tokenizer.do_lower_case:
                question_text = question_text.lower()
            query_tokens = self.tokenizer.tokenize(question_text)
            query_tokens = query_tokens[: self.max_query_length]

            para_text = example['paragraph_text']
            if self._hparams.dataset.tokenizer.do_lower_case:
                para_text = para_text.lower()
            para_tokens = self.tokenizer.tokenize(para_text)

            """
            For getting token to raw char matching:
            1) getting matching between token and tokenized text
            2) getting matching between raw text and tokenized text
            3) So, can get matching between token and raw
            """

            # char idx to token idx
            char2token_index = []
            # token start idx to char idx
            token2char_start_index = []
            # token end idx to char idx
            token2char_end_index = []
            char_idx = 0
            for i, token in enumerate(para_tokens):
                char_len = len(token.replace("##", ''))
                char2token_index.extend([i] * char_len)
                token2char_start_index.append(char_idx)
                char_idx += char_len
                token2char_end_index.append(char_idx - 1)

            # raw text ->(tokenizer)-> tokens ->(detokenizer)-> tokenized text
            tokenized_para_text = self.tokenizer.detokenizer(para_tokens)

            # matching between raw text and tokenized text
            N, M = len(para_text), len(tokenized_para_text)
            max_N, max_M = 1024, 1024
            if N > max_N or M > max_M:
                max_N = max(N, max_N)
                max_M = max(M, max_M)

            match_mapping, mismatch = self._generate_match_mapping(para_text, tokenized_para_text, N, M, max_N, max_M)

            # raw idx to tokenized char idx
            raw2tokenized_char_index = [None] * (N + 1)
            # tokenized char idx to raw idx
            tokenized2raw_char_index = [None] * (M + 1)
            i, j = N - 1, M - 1
            while i >= 0 and j >= 0:
                if (i, j) not in match_mapping:
                    break
                # if 324 == i or 353 == j:
                #     print()
                if match_mapping[(i, j)] == 2:
                    raw2tokenized_char_index[i] = j
                    tokenized2raw_char_index[j] = i
                    i, j = i - 1, j - 1
                elif match_mapping[(i, j)] == 1:
                    j = j - 1
                else:
                    i = i - 1

            # if para_text.startswith("安雅·罗素法"):
            #     print(tokenized2raw_char_index)

            if all(v is None for v in raw2tokenized_char_index) or mismatch:
                logger.warning("raw and tokenized paragraph mismatch detected for example: %s" % example['qas_id'])
                continue

            # token start idx to raw char idx
            token2char_raw_start_index = []
            # token end idx to raw char idx
            token2char_raw_end_index = []
            for idx in range(len(para_tokens)):
                # token char idx
                start_pos = token2char_start_index[idx]
                end_pos = token2char_end_index[idx]

                # raw char idx
                # try:
                raw_start_pos = self._convert_tokenized_index(tokenized2raw_char_index, start_pos, N, is_start=True)
                raw_end_pos = self._convert_tokenized_index(tokenized2raw_char_index, end_pos, N, is_start=False)
                # except:
                #     print(para_tokens[idx])
                #     print(''.join(para_tokens)[start_pos])
                #     print(''.join(para_tokens)[end_pos])


                # matching between token and raw char idx
                token2char_raw_start_index.append(raw_start_pos)
                token2char_raw_end_index.append(raw_end_pos)

            if not example['is_impossible']:
                # answer pos in raw text
                raw_start_char_pos, new_answer = self._improve_answer_start(para_text, example['orig_answer_text'], example['start_position'])
                example['orig_answer_text'] = new_answer
                # raw_start_char_pos = example['start_position']
                raw_end_char_pos = raw_start_char_pos + len(example['orig_answer_text']) - 1
                # answer pos in tokenized text
                tokenized_start_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_start_char_pos, is_start=True)
                tokenized_end_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_end_char_pos, is_start=False)
                # answer pos in tokens
                tokenized_start_token_pos = char2token_index[tokenized_start_char_pos]
                tokenized_end_token_pos = char2token_index[tokenized_end_char_pos]
                assert tokenized_start_token_pos <= tokenized_end_token_pos
            else:
                tokenized_start_token_pos = tokenized_end_token_pos = -1

            max_para_length = self._hparams.dataset.tokenizer.max_len - len(query_tokens) - 3

            total_para_length = len(para_tokens)

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            doc_spans = []
            para_start = 0
            while para_start < total_para_length:
                para_length = total_para_length - para_start
                if para_length > max_para_length:
                    para_length = max_para_length

                doc_spans.append({
                    "start": para_start,
                    "length": para_length
                })

                if para_start + para_length == total_para_length:
                    break

                para_start += min(para_length, self.doc_stride)

            for (doc_idx, doc_span) in enumerate(doc_spans):
                doc_token2char_raw_start_index = []
                doc_token2char_raw_end_index = []
                doc_token2doc_index = {}

                for i in range(doc_span['length']):
                    token_idx = doc_span["start"] + i

                    doc_token2char_raw_start_index.append(token2char_raw_start_index[token_idx])
                    doc_token2char_raw_end_index.append(token2char_raw_end_index[token_idx])

                    best_doc_idx = self._find_max_context(doc_spans, token_idx)
                    doc_token2doc_index[i] = (best_doc_idx == doc_idx)

                encode_info = \
                    self.tokenizer.encode(
                        query_tokens,
                        para_tokens[doc_span['start']: doc_span['start'] + doc_span['length']],
                        return_mask=True,
                        return_offset=True,
                        return_cls_index=True)
                input_ids, segment_ids, input_mask, p_mask, q_mask, offset, cls_idx = \
                    encode_info['input_ids'], encode_info['segment_ids'], encode_info['input_mask'], \
                    encode_info['b_mask'], encode_info['a_mask'], encode_info['b_offset'], encode_info['cls_index']

                p_mask[cls_idx] = 1
                start_position = None
                end_position = None
                is_impossible = example["is_impossible"]
                if not is_impossible:
                    # For training, if our document chunk does not contain an annotation, set default values.
                    doc_start = doc_span["start"]
                    doc_end = doc_start + doc_span["length"] - 1
                    if tokenized_start_token_pos < doc_start or tokenized_end_token_pos > doc_end:
                        start_position = cls_idx
                        end_position = cls_idx
                        is_impossible = 1
                    else:
                        start_position = tokenized_start_token_pos - doc_start + offset
                        end_position = tokenized_end_token_pos - doc_start + offset
                else:
                    start_position = cls_idx
                    end_position = cls_idx

                # if is_impossible == 1:
                #     continue

                item = {
                    "unique_id": unique_id,
                    "qas_id": example['qas_id'],
                    "question_text": question_text,
                    "context_text": para_text,
                    "answer_text": example["orig_answer_text"],
                    "all_answers": json.dumps(example["all_answers"]),
                    "doc_token2char_raw_start_index": json.dumps(doc_token2char_raw_start_index),
                    "doc_token2char_raw_end_index": json.dumps(doc_token2char_raw_end_index),
                    'doc_token2doc_index': json.dumps(doc_token2doc_index),
                    "input_ids": input_ids,
                    "token_type_ids": segment_ids,
                    "attention_mask": input_mask,
                    "p_mask": p_mask,
                    "start_position": str(start_position),
                    "end_position": str(end_position),
                    "is_impossible": is_impossible,
                    'offset': offset
                }

                if e_i == 0 and split == "train":
                    logger.info("*** Example ***")
                    logger.info(f"qas_id: {example['qas_id']}")
                    logger.info(f"unique_id: {unique_id}")
                    logger.info(f"question: {question_text}")
                    logger.info(f"context: {para_text}")
                    logger.info(f"answer: {example['orig_answer_text']}")
                    logger.info(f"qas_id: {example['qas_id']}")
                    logger.info(f"doc_idx: {doc_idx}")
                    logger.info(f"input_ids: {input_ids}")
                    logger.info(f"token_type_ids: {segment_ids}")
                    logger.info(f"attention_mask: {input_mask}")
                    logger.info(f"p_mask: {p_mask}")
                    logger.info(f"offset: {offset}")
                    logger.info(f"doc_token2char_raw_start_index: {doc_token2char_raw_start_index}")
                    logger.info(f"doc_token2char_raw_end_index: {doc_token2char_raw_end_index}")
                    logger.info(f"doc_token2doc_index: {doc_token2doc_index}")
                    logger.info(f"start_position: {start_position}")
                    logger.info(f"end_position: {end_position}")
                    logger.info(f"is_impossible: {is_impossible}")

                if start_position != 0 and end_position != 0 and split != 'test':
                    ccc = para_text[raw_start_char_pos: raw_end_char_pos + 1]
                    bbb = tokenized_para_text[tokenized_start_char_pos: tokenized_end_char_pos + 1]
                    aaa = para_tokens[tokenized_start_token_pos: tokenized_end_token_pos + 1]
                    ck_sp = start_position - offset
                    ck_ep = end_position - offset
                    raw_sp = doc_token2char_raw_start_index[ck_sp]
                    raw_ep = doc_token2char_raw_end_index[ck_ep]
                    answer_raw_span = para_text[raw_sp: raw_ep + 1]
                    if answer_raw_span != example['orig_answer_text'].lower():
                        logger.warning(f"Check Inputs: qas_id: {example['qas_id']}, unique_id: {unique_id}, orig_answer: {example['orig_answer_text']}, span_answer: {answer_raw_span}")

                # new_line = f"{json_dumps(item)}\n"
                # ouf.write(new_line)
                # logger.info(f"qas_id: {example['qas_id']}\tunique_id: {unique_id}")
                unique_id += 1
                yield item

        # return output_path

    def _read_next(self, data_path):
        with open(data_path, "r", encoding="utf8") as inf:
            data_list = json.load(inf)['data']
            for entry in data_list:
                for paragraph in entry['paragraphs']:
                    paragraph_text = paragraph['context']
                    for qa in paragraph['qas']:
                        qas_id = qa['id']
                        question_text = qa['question']

                        is_impossible = int(qa.get("is_impossible", len(qa['answers']) == 0))

                        if (len(qa["answers"]) == 0) and (not is_impossible):
                            raise ValueError(f"For training, each question should have exactly 1 answer for qas_id: {qas_id}.")
                        if not is_impossible:
                            answer = qa['answers'][0]
                            start_position = answer['answer_start']
                            orig_answer_text = answer['text']
                            all_answers = [ans['text'].lower() for ans in qa['answers']]
                        else:
                            start_position = -1
                            orig_answer_text = ""
                            all_answers = [orig_answer_text]

                        example = {
                            "qas_id": qas_id,
                            "question_text": question_text,
                            "paragraph_text": paragraph_text,
                            "orig_answer_text": orig_answer_text,
                            "start_position": start_position,
                            "is_impossible": is_impossible,
                            "all_answers": all_answers
                        }
                        yield example

    def _generate_match_mapping(self,
                                para_text,
                                tokenized_para_text,
                                N,
                                M,
                                max_N,
                                max_M):
        """Generate match mapping for raw and tokenized paragraph"""

        def _lcs_match(para_text,
                       tokenized_para_text,
                       N,
                       M,
                       max_N,
                       max_M,
                       max_dist):
            """longest common sub-sequence

            f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))

            unlike standard LCS, this is specifically optimized for the setting
            because the mismatch between sentence pieces and original text will be small
            """
            f = np.zeros((max_N, max_M), dtype=np.float32)
            g = {}

            for i in range(N):
                # if i == 324:
                #     print()
                for j in range(i - max_dist, i + max_dist):
                    # if j == 353:
                    #     print()
                    if j >= M or j < 0:
                        continue

                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0

                    raw_char = preprocess_text(para_text[i], self.tokenizer._hparams.do_lower_case, remove_space=False, keep_accents=True)
                    tokenized_char = tokenized_para_text[j]
                    if raw_char == tokenized_char and f_prev + 1 > f[i, j]:
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

            return f, g

        max_dist = abs(N - M) + 10
        for _ in range(2):
            lcs_matrix, match_mapping = _lcs_match(para_text, tokenized_para_text, N, M, max_N, max_M, max_dist)

            if lcs_matrix[N - 1, M - 1] > 0.8 * N:
                break

            max_dist *= 2

        mismatch = lcs_matrix[N - 1, M - 1] < 0.8 * N
        return match_mapping, mismatch

    def _convert_tokenized_index(self,
                                 index,
                                 pos,
                                 M=None,
                                 is_start=True):
        """Convert index for tokenized text"""
        if index[pos] is not None:
            return index[pos]

        N = len(index)
        rear = pos
        while rear < N - 1 and index[rear] is None:
            rear += 1

        front = pos
        while front > 0 and index[front] is None:
            front -= 1

        assert index[front] is not None or index[rear] is not None

        if index[front] is None:
            if index[rear] >= 1:
                if is_start:
                    return 0
                else:
                    return index[rear] - 1

            return index[rear]

        if index[rear] is None:
            if M is not None and index[front] < M - 1:
                if is_start:
                    return index[front] + 1
                else:
                    return M - 1

            return index[front]

        if is_start:
            if index[rear] > index[front] + 1:
                return index[front] + 1
            else:
                return index[rear]
        else:
            if index[rear] > index[front] + 1:
                return index[rear] - 1
            else:
                return index[front]

    def _find_max_context(self,
                          doc_spans,
                          token_idx):
        """Check if this is the 'max context' doc span for the token.
        Because of the sliding window approach taken to scoring documents, a single
        token can appear in multiple documents. E.g.
          Doc: the man went to the store and bought a gallon of milk
          Span A: the man went to the
          Span B: to the store and bought
          Span C: and bought a gallon of
          ...

        Now the word 'bought' will have two scores from spans B and C. We only
        want to consider the score with "maximum context", which we define as
        the *minimum* of its left and right context (the *sum* of left and
        right context will always be the same, of course).

        In the example the maximum context for 'bought' would be span C since
        it has 1 left context and 3 right context, while span B has 4 left context
        and 0 right context.
        """
        best_doc_score = None
        best_doc_idx = None
        for (doc_idx, doc_span) in enumerate(doc_spans):
            doc_start = doc_span["start"]
            doc_length = doc_span["length"]
            doc_end = doc_start + doc_length - 1
            if token_idx < doc_start or token_idx > doc_end:
                continue

            left_context_length = token_idx - doc_start
            right_context_length = doc_end - token_idx
            doc_score = min(left_context_length, right_context_length) + 0.01 * doc_length
            if best_doc_score is None or doc_score > best_doc_score:
                best_doc_score = doc_score
                best_doc_idx = doc_idx

        return best_doc_idx

    def _improve_answer_start(self, para_text, answer, raw_answer_start):
        answer = answer.lower().strip()
        real_start = para_text.find(answer)
        if real_start != -1:
            return real_start, answer
        else:
            return raw_answer_start, answer


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