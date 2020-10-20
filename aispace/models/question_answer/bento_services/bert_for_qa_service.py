# !/usr/bin/env python
# coding=utf-8
# @Time    : 2020/4/25 18:08
# @Author  : yingyuankai@aliyun.com
# @File    : bert_for_qa_service.py


__all__ = [
    "BertQAService"
]

import os, sys
from collections import defaultdict
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" * 4)))

from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TensorflowSavedModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

import numpy as np
from scipy.special import softmax, expit

from aispace.datasets.tokenizer import BertTokenizer
from aispace.utils.hparams import Hparams
from aispace.utils.str_utils import uuid_maker, preprocess_text, compute_md5_hash


@artifacts([
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('tokenizer'),
        PickleArtifact("hparams"),
    ])
@env(auto_pip_dependencies=True)
class BertQAService(BentoService):

    def preprocessing(self, parsed_json):
        unique_id = 100000
        for one_json in parsed_json:
            n_best_size = one_json.get('n_best_size', 5)
            max_answer_length = one_json.get("max_answer_length", 64)
            max_query_length = one_json.get("max_query_length", 64)
            doc_stride = one_json.get("doc_stride", 128)
            question_text = one_json.get("query", "")
            para_text = one_json.get("context", "")
            if question_text == "" or para_text == "":
                # unique_id = uuid_maker()
                print("[WARRING] query or context is empty!")
                item = {
                    "unique_id": unique_id,
                    "qas_id": unique_id,
                    "question_text": question_text,
                    "context_text": para_text,
                    'n_best_size': n_best_size,
                    'max_answer_length': max_answer_length
                }
                yield item

            qas_id = one_json.get('qas_id', compute_md5_hash(question_text + para_text))
            if self.artifacts.hparams.dataset.tokenizer.do_lower_case:
                question_text = question_text.lower()
            query_tokens = self.artifacts.tokenizer.tokenize(question_text)
            query_tokens = query_tokens[: max_query_length]

            if self.artifacts.hparams.dataset.tokenizer.do_lower_case:
                para_text = para_text.lower()
            para_tokens = self.artifacts.tokenizer.tokenize(para_text)

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

            tokenized_para_text = self.artifacts.tokenizer.detokenizer(para_tokens)

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

            if all(v is None for v in raw2tokenized_char_index) or mismatch:
                print("[WARRING] raw and tokenized paragraph mismatch detected")
                # unique_id = uuid_maker()
                item = {
                    "unique_id": unique_id,
                    "qas_id": qas_id,
                    "question_text": question_text,
                    "context_text": para_text,
                    'n_best_size': n_best_size,
                    'max_answer_length': max_answer_length
                }
                yield item

            # token start idx to raw char idx
            token2char_raw_start_index = []
            # token end idx to raw char idx
            token2char_raw_end_index = []
            for idx in range(len(para_tokens)):
                # token char idx
                start_pos = token2char_start_index[idx]
                end_pos = token2char_end_index[idx]

                # raw char idx
                raw_start_pos = self._convert_tokenized_index(tokenized2raw_char_index, start_pos, N, is_start=True)
                raw_end_pos = self._convert_tokenized_index(tokenized2raw_char_index, end_pos, N, is_start=False)

                # matching between token and raw char idx
                token2char_raw_start_index.append(raw_start_pos)
                token2char_raw_end_index.append(raw_end_pos)

            max_para_length = self.artifacts.hparams.dataset.tokenizer.max_len - len(query_tokens) - 3

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

                para_start += min(para_length, doc_stride)

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
                    self.artifacts.tokenizer.encode(
                        query_tokens,
                        para_tokens[doc_span['start']: doc_span['start'] + doc_span['length']],
                        return_mask=True,
                        return_offset=True,
                        return_cls_index=True)
                input_ids, segment_ids, input_mask, p_mask, q_mask, offset, cls_idx = \
                    encode_info['input_ids'], encode_info['segment_ids'], encode_info['input_mask'], \
                    encode_info['b_mask'], encode_info['a_mask'], encode_info['b_offset'], encode_info['cls_index']
                # unique_id = uuid_maker()

                item = {
                    "unique_id": unique_id,
                    "qas_id": qas_id,
                    "question_text": question_text,
                    "context_text": para_text,
                    "doc_token2char_raw_start_index": doc_token2char_raw_start_index,
                    "doc_token2char_raw_end_index": doc_token2char_raw_end_index,
                    'doc_token2doc_index': doc_token2doc_index,
                    "input_ids": input_ids,
                    "token_type_ids": segment_ids,
                    "attention_mask": input_mask,
                    "p_mask": p_mask,
                    'offset': offset,
                    'n_best_size': n_best_size,
                    'max_answer_length': max_answer_length
                }
                unique_id += 1
                yield item

    @api(JsonHandler)
    def qa_predict(self, parsed_json):
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": [], "p_mask": [], "unique_id": [], "start_position": []
        }
        no_answer_response = {
                        'predict_text': "",
                        'start_prob': 0.,
                        'end_prob': 0.,
                        'predict_score': 0.
                    }
        pre_input_data = self.preprocessing(parsed_json)

        qas_id_2_examples = defaultdict(list)
        unique_id_to_example = defaultdict()
        qas_ids = []
        for itm in pre_input_data:
            qas_ids.append(itm['qas_id'])
            if 'input_ids' not in itm:
                continue
            qas_id_2_examples[itm['qas_id']].append(itm)
            unique_id_to_example[itm['unique_id']] = itm
            input_data['input_ids'].append(itm['input_ids'])
            input_data['token_type_ids'].append(itm['token_type_ids'])
            input_data['attention_mask'].append(itm['attention_mask'])
            input_data['p_mask'].append(itm['p_mask'])
            # input_data['offset'].append(itm['offset'])
            # input_data['cls_idx'].append(itm['cls_idx'])
            input_data['unique_id'].append(itm['unique_id'])
            input_data['start_position'].append(0)

        if not input_data['input_ids']:
            print("[WARRING] Preprocessing some thing wrong!")
            return [no_answer_response]

        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        input_data['p_mask'] = tf.constant(input_data['p_mask'], name="p_mask")
        input_data['unique_id'] = tf.constant(input_data['unique_id'], dtype=tf.float32, name="unique_id")
        input_data['start_position'] = tf.constant(input_data['start_position'], name="start_position")

        start_top_res, end_top_res, unique_id_res = self.artifacts.model(input_data, training=False)
        start_top_log_prob, start_top_index = start_top_res.numpy()[:, :, 0], start_top_res.numpy()[:, :, 1].astype(np.int)  # [b, k]
        end_top_log_prob, end_top_index = end_top_res.numpy()[:, :, 0], end_top_res.numpy()[:, :, 1].astype(np.int)  # [b, k]
        unique_id_res = unique_id_res.numpy().astype(np.int)
        start_n_top, end_n_top = start_top_index.shape[-1], end_top_index.shape[-1]

        unique_id_2_result = {}
        for i in range(end_top_index.shape[0]):
            unique_id = unique_id_res[i]
            itm = {
                'unique_id': unique_id,
                'start_top_log_prob': start_top_log_prob[i],
                'start_top_index': start_top_index[i],
                'end_top_log_prob': end_top_log_prob[i],
                'end_top_index': end_top_index[i],
            }
            unique_id_2_result[unique_id] = itm

        answers = []

        no_answer_response = {
                        'predict_text': "",
                        'span_start': 0,
                        'start_prob': 0.,
                        'span_end': 0,
                        'end_prob': 0.,
                        'predict_score': 0.
                    }

        for qas_id in qas_ids:
            examples = qas_id_2_examples.get(qas_id, [])
            if not examples:
                answers.append(no_answer_response)
                continue
            max_answer_length, n_best_size = examples[0].get('max_answer_length'), examples[0].get('n_best_size')
            example_all_predicts = []
            for example in examples:
                cur_unique_id = example['unique_id']
                if cur_unique_id not in unique_id_2_result:
                    continue
                cur_result = unique_id_2_result.get(cur_unique_id)
                cur_start_top_log_prob = cur_result['start_top_log_prob']
                cur_start_top_index = cur_result['start_top_index']

                cur_end_top_log_prob = cur_result['end_top_log_prob']
                cur_end_top_index = cur_result['end_top_index']

                cur_p_mask = example['p_mask']
                for i in range(start_n_top):
                    start_prob = cur_start_top_log_prob[i]
                    start_index = cur_start_top_index[i]

                    if not cur_p_mask[start_index]:
                        continue

                    for j in range(end_n_top):
                        end_prob = cur_end_top_log_prob[j]
                        end_index = cur_end_top_index[j]

                        if not cur_p_mask[end_index]:
                            continue

                        answer_length = end_index - start_index + 1
                        if end_index < start_index or answer_length > max_answer_length:
                            continue

                        itm = {
                            'unique_id': cur_unique_id,
                            'start_prob': start_prob,
                            'start_index': start_index,
                            'end_prob': end_prob,
                            'end_index': end_index,
                            'predict_score': np.log(start_prob) + np.log(end_prob)
                        }
                        example_all_predicts.append(itm)

            example_all_predicts.sort(key=lambda s: s['predict_score'], reverse=True)

            example_top_predicts = []
            is_visited = set()
            for example_predict in example_all_predicts:
                if len(example_top_predicts) >= n_best_size:
                    break
                example_feature = unique_id_to_example[example_predict['unique_id']]
                predict_start = example_feature['doc_token2char_raw_start_index'][
                    example_predict['start_index'] - example_feature['offset']]
                predict_end = example_feature['doc_token2char_raw_end_index'][
                    example_predict['end_index'] - example_feature['offset']]
                predict_text = example_feature['context_text'][predict_start: predict_end + 1].strip()

                if predict_text in is_visited:
                    continue

                itm = {
                    'predict_text': predict_text,
                    'span_start': predict_start,
                    'start_prob': example_predict['start_prob'],
                    'span_end': predict_end,
                    'end_prob': example_predict['end_prob'],
                    'predict_score': example_predict['predict_score']
                }
                example_top_predicts.append(itm)

            if len(example_top_predicts) == 0:
                example_top_predicts.append(
                    no_answer_response
                )

            example_best_predict = example_top_predicts[0]
            answers.append(example_best_predict)
        return answers

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

                    raw_char = preprocess_text(para_text[i], self.artifacts.hparams.dataset.tokenizer.do_lower_case, remove_space=False, keep_accents=True)
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
        if lcs_matrix[N - 1, M - 1] == min(M, N):
            mismatch = False
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