# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-02 21:14
# @Author  : yingyuankai@aliyun.com
# @File    : metrics_utils.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    "ConfusionMatrix",
    "NEREvaluator"
]

import logging
from collections import OrderedDict
from collections import namedtuple
from copy import deepcopy

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


class ConfusionMatrix:
    def __init__(self, conditions, predictions, labels=None,
                 sample_weight=None):
        # assert (len(predictions) == len(conditions))
        min_length = min(len(predictions), len(conditions))
        self.predictions = predictions[:min_length]
        self.conditions = conditions[:min_length]

        if labels is not None:
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
            self.idx2label = {idx: label for idx, label in enumerate(labels)}
            labels = list(range(len(labels)))
        else:
            self.label2idx = {str(label): idx for idx, label in
                              enumerate(np.unique(
                                  [self.predictions, self.conditions]))}
            self.idx2label = {idx: str(label) for idx, label in
                              enumerate(np.unique(
                                  [self.predictions, self.conditions]))}

        self.label_names = [v for k, v in sorted(self.idx2label.items(), key=lambda s: s[0])]
        self.cm = confusion_matrix(self.conditions,
                                   self.predictions,
                                   labels=labels,
                                   sample_weight=sample_weight)

        self.sum_predictions = np.sum(self.cm, axis=0)
        self.sum_conditions = np.sum(self.cm, axis=1)
        self.all = np.sum(self.cm)

    def confusion_matrix_visual(self):
        pt = PrettyTable([""] + self.label_names)
        label_len = len(self.label_names)
        for i in range(label_len):
            new_row = [self.label_names[i]]
            for j in range(label_len):
                new_row.append(self.cm[i, j])
            pt.add_row(new_row)
        return pt.get_string()

    def label_to_idx(self, label):
        return self.label2idx[label]

    def true_positives(self, idx):
        return self.cm[idx, idx]

    def true_negatives(self, idx):
        return self.all - self.sum_predictions[idx] - self.sum_conditions[
            idx] + self.true_positives(idx)

    def false_positives(self, idx):
        return self.sum_predictions[idx] - self.true_positives(idx)

    def false_negatives(self, idx):
        return self.sum_conditions[idx] - self.true_positives(idx)

    def true_positive_rate(self, idx):
        nom = self.true_positives(idx)
        den = self.sum_conditions[idx]
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def true_negative_rate(self, idx):
        nom = tn = self.true_negatives(idx)
        den = tn + self.false_positives(idx)
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def positive_predictive_value(self, idx):
        nom = self.true_positives(idx)
        den = self.sum_predictions[idx]
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def negative_predictive_value(self, idx):
        nom = tn = self.true_negatives(idx)
        den = tn + self.false_negatives(idx)
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def false_negative_rate(self, idx):
        return 1.0 - self.true_positive_rate(idx)

    def false_positive_rate(self, idx):
        return 1.0 - self.true_negative_rate(idx)

    def false_discovery_rate(self, idx):
        return 1.0 - self.positive_predictive_value(idx)

    def false_omission_rate(self, idx):
        return 1.0 - self.negative_predictive_value(idx)

    def accuracy(self, idx):
        nom = self.true_positives(idx) + self.true_negatives(idx)
        den = self.all
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def precision(self, idx):
        return self.positive_predictive_value(idx)

    def recall(self, idx):
        return self.true_positive_rate(idx)

    def fbeta_score(self, beta, idx):
        beta_2 = np.power(beta, 2)
        precision = self.precision(idx)
        recall = self.recall(idx)
        nom = (1 + beta_2) * precision * recall
        den = (beta_2 * precision) + recall
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def f1_score(self, idx):
        return self.fbeta_score(1, idx)

    def sensitivity(self, idx):
        return self.true_positive_rate(idx)

    def specificity(self, idx):
        return self.true_negative_rate(idx)

    def hit_rate(self, idx):
        return self.true_positive_rate(idx)

    def miss_rate(self, idx):
        return self.false_negative_rate(idx)

    def fall_out(self, idx):
        return self.false_positive_rate(idx)

    def matthews_correlation_coefficient(self, idx):
        tp = self.true_positives(idx)
        tn = self.true_negatives(idx)
        fp = self.false_positives(idx)
        fn = self.false_negatives(idx)
        nom = tp * tn - fp * fn
        den = np.sqrt(tp + fp) * np.sqrt(tp + fn) * np.sqrt(tn + fp) * np.sqrt(tn + fn)
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def informedness(self, idx):
        return self.true_positive_rate(idx) + self.true_negative_rate(idx) - 1

    def markedness(self, idx):
        return self.positive_predictive_value(
            idx) + self.negative_predictive_value(idx) - 1

    def overall_accuracy(self):
        return metrics.accuracy_score(self.conditions, self.predictions)

    def avg_precision(self, average='macro'):
        return metrics.precision_score(self.conditions, self.predictions,
                                       average=average)

    def avg_recall(self, average='macro'):
        return metrics.recall_score(self.conditions, self.predictions,
                                    average=average)

    def avg_f1_score(self, average='macro'):
        return metrics.f1_score(self.conditions, self.predictions,
                                average=average)

    def avg_fbeta_score(self, beta, average='macro'):
        return metrics.fbeta_score(self.conditions, self.predictions, beta=beta,
                                   average=average)

    def kappa_score(self):
        return metrics.cohen_kappa_score(self.conditions, self.predictions)

    def class_stats(self, idx):
        return {
            'true_positives': self.true_positives(idx),
            'true_negatives': self.true_negatives(idx),
            'false_positives': self.false_positives(idx),
            'false_negatives': self.false_negatives(idx),
            'true_positive_rate': self.true_positive_rate(idx),
            'true_negative_rate': self.true_negative_rate(idx),
            'positive_predictive_value': self.positive_predictive_value(idx),
            'negative_predictive_value': self.negative_predictive_value(idx),
            'false_negative_rate': self.false_negative_rate(idx),
            'false_positive_rate': self.false_positive_rate(idx),
            'false_discovery_rate': self.false_discovery_rate(idx),
            'false_omission_rate': self.false_omission_rate(idx),
            'accuracy': self.accuracy(idx),
            'precision': self.precision(idx),
            'recall': self.recall(idx),
            'f1_score': self.f1_score(idx),
            'sensitivity': self.sensitivity(idx),
            'specificity': self.specificity(idx),
            'hit_rate': self.hit_rate(idx),
            'miss_rate': self.miss_rate(idx),
            'fall_out': self.fall_out(idx),
            'matthews_correlation_coefficient': self.matthews_correlation_coefficient(
                idx),
            'informedness': self.informedness(idx),
            'markedness': self.markedness(idx)
        }

    def per_class_stats(self):
        stats = OrderedDict()
        for idx in sorted(self.idx2label.keys()):
            stats[self.idx2label[idx]] = self.class_stats(idx)
        return stats

    def stats(self):
        return {
            'overall_accuracy': self.overall_accuracy(),
            'avg_precision_macro': self.avg_precision(average='macro'),
            'avg_recall_macro': self.avg_recall(average='macro'),
            'avg_f1_score_macro': self.avg_f1_score(average='macro'),
            'avg_precision_micro': self.avg_precision(average='micro'),
            'avg_recall_micro': self.avg_recall(average='micro'),
            'avg_f1_score_micro': self.avg_f1_score(average='micro'),
            'avg_precision_weighted': self.avg_precision(average='micro'),
            'avg_recall_weighted': self.avg_recall(average='micro'),
            'avg_f1_score_weighted': self.avg_f1_score(average='weighted'),
            'kappa_score': self.kappa_score()
        }


Entity = namedtuple("Entity", "e_type start_offset end_offset")


class NEREvaluator:

    def __init__(self, true, pred, tags):
        """
        Ref: http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        """

        if len(true) != len(pred):
            raise ValueError("Number of predicted documents does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.

        self.metrics_results = {
            'correct': 0,
            'incorrect': 0,
            'partial': 0,
            'missed': 0,
            'spurious': 0,
            'possible': 0,
            'actual': 0,
            'precision': 0,
            'recall': 0,
        }

        # Copy results dict to cover the four schemes.

        self.results = {
            'strict': deepcopy(self.metrics_results),
            'ent_type': deepcopy(self.metrics_results),
            'partial': deepcopy(self.metrics_results),
            'exact': deepcopy(self.metrics_results),
        }

        # Create an accumulator to store results

        self.evaluation_agg_entities_type = {e: deepcopy(self.results) for e in tags}

    def evaluate(self):

        logger.info(
            "Imported %s predictions for %s true examples",
            len(self.pred), len(self.true)
        )

        for true_ents, pred_ents in zip(self.true, self.pred):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError("Prediction length does not match true example length")

            # Compute results for one message

            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents),
                collect_named_entities(pred_ents),
                self.tags
            )

            # Cycle through each result and accumulate

            # TODO: Combine these loops below:

            for eval_schema in self.results:

                for metric in self.results[eval_schema]:
                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

            # Calculate global precision and recall

            self.results = compute_precision_recall_wrapper(self.results)

            # Aggregate results by entity type

            for e_type in self.tags:

                for eval_schema in tmp_agg_results[e_type]:

                    for metric in tmp_agg_results[e_type][eval_schema]:
                        self.evaluation_agg_entities_type[e_type][eval_schema][metric] += \
                        tmp_agg_results[e_type][eval_schema][metric]

                # Calculate precision recall at the individual entity level

                self.evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                    self.evaluation_agg_entities_type[e_type])

        return self.results, self.evaluation_agg_entities_type


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, tags):
    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0, 'precision': 0, 'recall': 0}

    # overall results

    evaluation = {
        'strict': deepcopy(eval_metrics),
        'ent_type': deepcopy(eval_metrics),
        'partial': deepcopy(eval_metrics),
        'exact': deepcopy(eval_metrics)
    }

    # results by entity type

    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in tags}

    # keep track of entities that overlapped

    true_which_overlapped_with_pred = []

    # Subset into only the tags that we are interested in.
    # NOTE: we remove the tags we don't want from both the predicted and the
    # true entities. This covers the two cases where mismatches can occur:
    #
    # 1) Where the model predicts a tag that is not present in the true data
    # 2) Where there is a tag in the true data that the model is not capable of
    # predicting.

    true_named_entities = [ent for ent in true_named_entities if ent.e_type in tags]
    pred_named_entities = [ent for ent in pred_named_entities if ent.e_type in tags]

    # go through each predicted named-entity

    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities

            for true in true_named_entities:

                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset \
                        and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True

                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        found_overlap = True

                        break

                    # Scenario VI: Entities overlap, but the entity type is
                    # different.

                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True

                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:

                # Overall results

                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # Aggregated by entity type results

                # NOTE: when pred.e_type is not found in tags
                # or when it simply does not appear in the test set, then it is
                # spurious, but it is not clear where to assign it at the tag
                # level. In this case, it is applied to all target_tags
                # found in this example. This will mean that the sum of the
                # evaluation_agg_entities will not equal evaluation.

                for true in tags:
                    evaluation_agg_entities_type[true]['strict']['spurious'] += 1
                    evaluation_agg_entities_type[true]['ent_type']['spurious'] += 1
                    evaluation_agg_entities_type[true]['partial']['spurious'] += 1
                    evaluation_agg_entities_type[true]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:
            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: compute_precision_recall(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: compute_precision_recall(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}

    return results


def roc_curve(conditions, prediction_scores, pos_label=None,
              sample_weight=None):
    return metrics.roc_curve(conditions, prediction_scores, pos_label,
                             sample_weight)


def roc_auc_score(conditions, prediction_scores, average='micro',
                  sample_weight=None):
    try:
        return metrics.roc_auc_score(conditions, prediction_scores, average,
                                     sample_weight)
    except ValueError as ve:
        logger.info(ve)


def precision_recall_curve(conditions, prediction_scores, pos_label=None,
                           sample_weight=None):
    return metrics.precision_recall_curve(conditions, prediction_scores,
                                          pos_label, sample_weight)


def average_precision_score(conditions, prediction_scores, average='micro',
                            sample_weight=None):
    # average == [micro, macro, sampled, weidhted]
    return metrics.average_precision_score(conditions, prediction_scores,
                                           average=average,
                                           sample_weight=sample_weight)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='This script trains and tests a model.')
#     parser.add_argument('gold_standard', help='file containing gold standars')
#     parser.add_argument(PREDICTIONS, help='file containing predictions')
#     parser.add_argument('output_fp', help='output file')
#     args = parser.parse_args()
#
#     hdf5_data = h5py.File(args.gold_standard, 'r')
#     split = hdf5_data['split'].value
#     column = hdf5_data['macros'].value
#     hdf5_data.close()
#     conditions = column[split == 2]  # ground truth
#
#     predictions = np.load(args.predictions)
#
#     confusion_matrix = ConfusionMatrix(predictions, conditions)
#
#     results = load_json(args.output_fp)
#     results['confusion_matrix_stats'] = {
#         'confusion_matrix': confusion_matrix.cm.tolist(),
#         'overall_stats': confusion_matrix.stats(),
#         'per_class_stats': confusion_matrix.per_class_stats()
#     }
#     save_json(args.output_fp, results)
