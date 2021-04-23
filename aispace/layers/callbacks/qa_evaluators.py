# -*- coding: utf-8 -*-
# @Time    : 2020-07-30 15:06
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : qa_evaluators.py


import os
import logging
import numpy as np
import tensorflow as tf
import json
from pprint import pprint
from collections import defaultdict

from aispace.utils.eval_utils import calc_em_score, calc_f1_score
from aispace.utils.io_utils import save_json
from aispace.utils.print_utils import print_boxed

__all__ = [
    'EvaluatorForQaSimple',
    'EvaluatorForQaWithImpossible'
]

logger = logging.getLogger(__name__)


class EvaluatorForQaSimple(tf.keras.callbacks.Callback):
    """
    start_top_log_prob and end_top_log_prob's shape is [batch, k]
    ref: https://keras.io/examples/nlp/text_extraction_with_bert/
    """

    def __init__(self, validation_dataset, validation_steps, test_dataset, test_steps, report_dir, max_answer_length=64, n_best_size=5):
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps
        self.test_dataset = test_dataset
        self.test_steps = test_steps
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.report_dir = report_dir

    def on_epoch_end(self, epoch, logs=None):
        new_logs = self.eval_process(self.validation_dataset, self.validation_steps)
        logs = logs or {}
        logs.update(new_logs)
        print(f"Epoch: {epoch + 1}, val_f1_score: {logs['val_f1_score']:.4f}, val_em_score: {logs['val_em_score']:.4f}, "
                    f"val_f1_em_avg_score: {logs['val_f1_em_avg_score']:.4f}")

    def on_train_end(self, logs=None):
        logger.info("Start Evaluate.")
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        new_logs = self.eval_process(self.test_dataset, self.test_steps)
        save_json(os.path.join(self.report_dir, 'performance.json'), new_logs)
        print_boxed(f"Question Answer Evaluation")
        pprint(new_logs)
        logger.info(f"Save question answer reports in {self.report_dir}")

    def eval_process(self, dataset, n_steps=None):
        f1 = 0
        em = 0
        total_count = 0
        skip_count = 0

        start_top_res, end_top_res, unique_id_res = self.model.predict(dataset, steps=n_steps)
        start_top_log_prob, start_top_index = start_top_res[:, :, 0], start_top_res[:, :, 1].astype(np.int)  # [b, k]
        end_top_log_prob, end_top_index = end_top_res[:, :, 0], end_top_res[:, :, 1].astype(np.int)  # [b, k]
        unique_id_res = unique_id_res.astype(np.int)
        # predict results
        results = {}
        for i in range(end_top_index.shape[0]):
            unique_id = unique_id_res[i][0]
            itm = {
                'unique_id': unique_id,
                'start_top_log_prob': start_top_log_prob[i],
                'start_top_index': start_top_index[i],
                'end_top_log_prob': end_top_log_prob[i],
                'end_top_index': end_top_index[i],
            }
            results[unique_id] = itm

        # raw inputs
        start_n_top, end_n_top = start_top_index.shape[-1], end_top_index.shape[-1]
        qas_id_to_examples = defaultdict(list)
        unique_id_to_examples = {}
        for idx, (inputs, outputs) in enumerate(dataset):
            if n_steps is not None and idx >= n_steps:
                break
            unique_ids = inputs['unique_id'].numpy().astype(np.int).tolist()
            offsets = inputs['offset'].numpy().astype(np.int).tolist()
            qas_ids = inputs['qas_id'].numpy().astype(str).tolist()
            doc_token2char_raw_start_indexs = inputs['doc_token2char_raw_start_index'].numpy().astype(str).tolist()
            doc_token2char_raw_end_indexs = inputs['doc_token2char_raw_end_index'].numpy().astype(str).tolist()
            doc_token2doc_indexs = inputs['doc_token2doc_index'].numpy().astype(str).tolist()
            all_answers = inputs['all_answers'].numpy().astype(str).tolist()
            answer_texts = inputs['answer_text'].numpy().tolist()
            context_texts = inputs['context_text'].numpy().tolist()
            question_texts = inputs['question_text'].numpy().tolist()
            is_impossibles = inputs['is_impossible'].numpy().tolist()
            p_masks = inputs['p_mask'].numpy().astype(np.int).tolist()

            for t in range(len(unique_ids)):
                itm = {
                    'unique_id': unique_ids[t],
                    'qas_id': qas_ids[t],
                    'question_text': question_texts[t].decode("utf8"),
                    'context_text': context_texts[t].decode("utf8"),
                    'answer_text': answer_texts[t].decode("utf8"),
                    'all_answers': json.loads(all_answers[t]),
                    'doc_token2char_raw_start_index': json.loads(doc_token2char_raw_start_indexs[t]),
                    'doc_token2char_raw_end_index': json.loads(doc_token2char_raw_end_indexs[t]),
                    'doc_token2doc_index': json.loads(doc_token2doc_indexs[t]),
                    'is_impossible': is_impossibles[t],
                    'p_mask': p_masks[t],
                    'offset': offsets[t]
                }
                unique_id_to_examples[unique_ids[t]] = itm
                qas_id_to_examples[qas_ids[t]].append(itm)

        for qas_id, examples in qas_id_to_examples.items():
            example_all_predicts = []
            answers = set()
            for example in examples:
                cur_unique_id = example['unique_id']
                if cur_unique_id not in results:
                    continue
                if example['is_impossible'] == 1:
                    continue
                # if example['answer_text'] not in answers:
                #     answers.append(example['answer_text'])
                answers |= set(example['all_answers'])
                cur_result = results.get(cur_unique_id)
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
                        if end_index < start_index or answer_length > self.max_answer_length:
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

            if len(answers) != 0:
                total_count += 1
            else:
                skip_count += 1
                continue

            example_all_predicts.sort(key=lambda s: s['predict_score'], reverse=True)

            example_top_predicts = []
            is_visited = set()
            for example_predict in example_all_predicts:
                if len(example_top_predicts) >= self.n_best_size:
                    break
                example_feature = unique_id_to_examples[example_predict['unique_id']]
                if example_predict['start_index'] - example_feature['offset'] < 0 or example_predict['end_index'] - example_feature['offset'] < 0:
                    predict_text = ""
                else:
                    predict_start = example_feature['doc_token2char_raw_start_index'][
                        example_predict['start_index'] - example_feature['offset']]
                    predict_end = example_feature['doc_token2char_raw_end_index'][
                        example_predict['end_index'] - example_feature['offset']]
                    predict_text = example_feature['context_text'][predict_start: predict_end + 1].strip()

                if predict_text in is_visited:
                    continue

                is_visited.add(predict_text)
                itm = {
                    'predict_text': predict_text,
                    'start_prob': example_predict['start_prob'],
                    'end_prob': example_predict['end_prob'],
                    'predict_score': example_predict['predict_score']
                }
                example_top_predicts.append(itm)

            if len(example_top_predicts) == 0:
                example_top_predicts.append(
                    {
                        'predict_text': "",
                        'start_prob': 0.,
                        'end_prob': 0.,
                        'predict_score': 0.
                    }
                )

            example_best_predict = example_top_predicts[0]

            cur_f1 = calc_f1_score(list(answers), example_best_predict['predict_text'])
            cur_em = calc_em_score(list(answers), example_best_predict['predict_text'])

            f1 += cur_f1
            em += cur_em
            # debug
            # if cur_f1 != 0 or cur_em != 0:
            #     example_output = {}
            #     example_output.update(example_best_predict)
            #     example_output['question'] = examples[0]['question_text']
            #     example_output['answer'] = answers
            #     example_output['f1'] = cur_f1
            #     example_output['em'] = cur_em
            #     print(example_output)

        # total_count = len(qas_id_to_examples)
        f1_score = f1 / total_count
        em_score = em / total_count
        logs = {}
        logs['skip_count'] = skip_count
        logs['total'] = total_count
        logs['val_f1_score'] = f1_score
        logs['val_em_score'] = em_score
        logs['val_f1_em_avg_score'] = (em_score + f1_score) / 2.
        return logs


class EvaluatorForQaWithImpossible(tf.keras.callbacks.Callback):
    """
    start_top_log_prob and end_top_log_prob's shape is [batch, k, k]
    ref: https://keras.io/examples/nlp/text_extraction_with_bert/
    """

    def __init__(self, validation_dataset, validation_steps, test_dataset, test_steps, report_dir, max_answer_length=64, n_best_size=5):
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps
        self.test_dataset = test_dataset
        self.test_steps = test_steps
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.report_dir = report_dir

    def on_epoch_end(self, epoch, logs=None):
        new_logs = self.eval_process(self.validation_dataset, self.validation_steps)
        logs = logs or {}
        logs.update(new_logs)
        print(f"\nEpoch: {epoch + 1}, val_f1_score: {logs['val_f1_score']:.4f}, val_em_score: {logs['val_em_score']:.4f}, "
                    f"val_f1_em_avg_score: {logs['val_f1_em_avg_score']:.4f}")

    def on_train_end(self, logs=None):
        logger.info("Start Evaluate.")
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        new_logs = self.eval_process(self.test_dataset, self.test_steps)
        save_json(os.path.join(self.report_dir, 'performance.json'), new_logs)
        print_boxed(f"Question Answer Evaluation")
        pprint(new_logs)
        logger.info(f"Save question answer reports in {self.report_dir}")

    def eval_process(self, dataset, n_steps=None):
        f1 = 0
        em = 0
        total_count = 0
        skip_count = 0

        start_top_res, end_top_res, answer_prob, unique_id_res = self.model.predict(dataset, steps=n_steps)
        start_top_log_prob, start_top_index = start_top_res[:, :, 0], start_top_res[:, :, 1].astype(np.int)  # [b, k]
        end_top_log_prob, end_top_index = end_top_res[:, :, :, 0], end_top_res[:, :, :, 1].astype(np.int)  # [b, k, k]
        unique_id_res = unique_id_res.astype(np.int)
        # predict results
        results = {}
        for i in range(end_top_index.shape[0]):
            unique_id = unique_id_res[i][0]
            itm = {
                'unique_id': unique_id,
                'start_top_log_prob': start_top_log_prob[i],
                'start_top_index': start_top_index[i],
                'end_top_log_prob': end_top_log_prob[i],
                'end_top_index': end_top_index[i],
                'is_impossible_prob': answer_prob[i][0]
            }
            results[unique_id] = itm

        # raw inputs
        start_n_top, end_n_top = end_top_index.shape[1:]
        qas_id_to_examples = defaultdict(list)
        unique_id_to_examples = {}
        for idx, (inputs, outputs) in enumerate(dataset):
            if n_steps is not None and idx >= n_steps:
                break
            unique_ids = inputs['unique_id'].numpy().astype(np.int).tolist()
            offsets = inputs['offset'].numpy().astype(np.int).tolist()
            qas_ids = inputs['qas_id'].numpy().astype(str).tolist()
            doc_token2char_raw_start_indexs = inputs['doc_token2char_raw_start_index'].numpy().astype(str).tolist()
            doc_token2char_raw_end_indexs = inputs['doc_token2char_raw_end_index'].numpy().astype(str).tolist()
            doc_token2doc_indexs = inputs['doc_token2doc_index'].numpy().astype(str).tolist()
            all_answers = inputs['all_answers'].numpy().astype(str).tolist()
            answer_texts = inputs['answer_text'].numpy().tolist()
            context_texts = inputs['context_text'].numpy().tolist()
            question_texts = inputs['question_text'].numpy().tolist()
            is_impossibles = inputs['is_impossible'].numpy().tolist()
            p_masks = inputs['p_mask'].numpy().astype(np.int).tolist()

            for t in range(len(unique_ids)):
                itm = {
                    'unique_id': unique_ids[t],
                    'qas_id': qas_ids[t],
                    'question_text': question_texts[t].decode("utf8"),
                    'context_text': context_texts[t].decode("utf8"),
                    'answer_text': answer_texts[t].decode("utf8"),
                    'all_answers': json.loads(all_answers[t]),
                    'doc_token2char_raw_start_index': json.loads(doc_token2char_raw_start_indexs[t]),
                    'doc_token2char_raw_end_index': json.loads(doc_token2char_raw_end_indexs[t]),
                    'doc_token2doc_index': json.loads(doc_token2doc_indexs[t]),
                    'is_impossible': is_impossibles[t],
                    'p_mask': p_masks[t],
                    'offset': offsets[t]
                }
                unique_id_to_examples[unique_ids[t]] = itm
                qas_id_to_examples[qas_ids[t]].append(itm)

        for qas_id, examples in qas_id_to_examples.items():
            example_all_predicts = []
            answers = set()
            for example in examples:
                cur_unique_id = example['unique_id']
                if cur_unique_id not in results:
                    continue
                if example['is_impossible'] == 1:
                    continue
                # if example['answer_text'] not in answers:
                #     answers.append(example['answer_text'])
                answers |= set(example['all_answers'])
                cur_result = results.get(cur_unique_id)
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
                        end_prob = cur_end_top_log_prob[i, j]
                        end_index = cur_end_top_index[i, j]

                        if not cur_p_mask[end_index]:
                            continue

                        answer_length = end_index - start_index + 1
                        if end_index < start_index or answer_length > self.max_answer_length:
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

            if len(answers) != 0:
                total_count += 1
            else:
                skip_count += 1
                continue

            example_all_predicts.sort(key=lambda s: s['predict_score'], reverse=True)

            example_top_predicts = []
            is_visited = set()
            for example_predict in example_all_predicts:
                if len(example_top_predicts) >= self.n_best_size:
                    break

                example_feature = unique_id_to_examples[example_predict['unique_id']]
                predict_start = example_feature['doc_token2char_raw_start_index'][
                    example_predict['start_index'] - example_feature['offset']]
                predict_end = example_feature['doc_token2char_raw_end_index'][
                    example_predict['end_index'] - example_feature['offset']]
                predict_text = example_feature['context_text'][predict_start: predict_end + 1].strip()

                if predict_text in is_visited:
                    continue

                is_visited.add(predict_text)
                itm = {
                    'predict_text': predict_text,
                    'start_prob': example_predict['start_prob'],
                    'end_prob': example_predict['end_prob'],
                    'predict_score': example_predict['predict_score']
                }
                example_top_predicts.append(itm)

            if len(example_top_predicts) == 0:
                example_top_predicts.append(
                    {
                        'predict_text': "",
                        'start_prob': 0.,
                        'end_prob': 0.,
                        'predict_score': 0.
                    }
                )

            example_best_predict = example_top_predicts[0]

            cur_f1 = calc_f1_score(list(answers), example_best_predict['predict_text'])
            cur_em = calc_em_score(list(answers), example_best_predict['predict_text'])

            f1 += cur_f1
            em += cur_em
            # debug
            # if cur_f1 != 0 or cur_em != 0:
            #     example_output = {}
            #     example_output.update(example_best_predict)
            #     example_output['question'] = examples[0]['question_text']
            #     example_output['answer'] = answers
            #     example_output['f1'] = cur_f1
            #     example_output['em'] = cur_em
            #     print(example_output)

        # total_count = len(qas_id_to_examples)
        f1_score = f1 / total_count
        em_score = em / total_count

        logs = {}
        logs['skip_count'] = skip_count
        logs['total'] = total_count
        logs['val_f1_score'] = f1_score
        logs['val_em_score'] = em_score
        logs['val_f1_em_avg_score'] = (em_score + f1_score) / 2.

        return logs

