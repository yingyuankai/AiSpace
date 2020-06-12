# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 16:33
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : eval_utils.py


import os
from tqdm import tqdm
import logging
import json
from copy import deepcopy
from pprint import pprint
import tensorflow as tf

logger = logging.getLogger(__name__)

from aispace.utils.io_utils import save_json
from aispace.utils.hparams import Hparams
from aispace.utils.metrics_utils import ConfusionMatrix, NEREvaluator
from aispace.constants import *
from aispace.utils.print_utils import print_boxed
from aispace.utils.builder_utils import build_model, load_dataset


# TODO more general
def evaluation(hparams: Hparams, model=None, test_dataset=None):
    """Evaluate the model and build report according to different task.

    :param model:
    :param test_dataset:
    :param hparams:
    :return:
    """
    logger.info("Start Evaluate.")
    output_hparams = deepcopy(hparams.dataset.outputs)
    if model is None:
        # build model
        (model,) = build_model(hparams, return_losses=False, return_metrics=False, return_optimizer=False)
        if not os.path.exists(hparams.get_model_filename() + ".index"):
            logger.warning(f"Model from {hparams.get_model_filename()} is not exists, load nothing!")
        else:
            logger.info(f"Load model weights from {hparams.get_model_filename()}")
            model.load_weights(hparams.get_model_filename())

    if test_dataset is None:
        test_dataset = next(load_dataset(hparams, ret_train=False, ret_dev=False, ret_info=False))[0]

    # prediction
    # print(model.evaluate(test_dataset))
    for inputs, outputs in tqdm(test_dataset):
        model_outputs = model.predict(inputs)
        if not isinstance(model_outputs, (tuple, list)):
            model_outputs = (model_outputs,)
        for idx, one_output_hparam in enumerate(output_hparams):
            if "ground_truth" not in one_output_hparam:
                one_output_hparam["ground_truth"] = []
            if "predictions" not in one_output_hparam:
                one_output_hparam['predictions'] = []
            prediction_output = model_outputs[idx]
            tmp_name = one_output_hparam.name
            tmp_type = one_output_hparam.type
            tmp_ground_truth = outputs[tmp_name]
            if tmp_type in [CLASSLABEL, LIST_OF_CLASSLABEL, LIST_OF_INT]:
                if tmp_type in [LIST_OF_INT]:
                    tmp_tg = tf.argmax(tmp_ground_truth, -1)
                else:
                    tmp_tg = tmp_ground_truth
                if one_output_hparam.task == NER: # [[sent1], [sent2]]
                    one_output_hparam.ground_truth.extend(tmp_tg.numpy().tolist())
                    tmp_predictions = tf.argmax(prediction_output, -1).numpy().tolist()
                    one_output_hparam.predictions.extend(tmp_predictions)
                else: # [1, 0, 1, ...]
                    one_output_hparam.ground_truth.extend(tmp_tg.numpy().reshape(-1).tolist())
                    tmp_predictions = tf.argmax(prediction_output, -1).numpy().reshape(-1).tolist()
                    one_output_hparam.predictions.extend(tmp_predictions)

    # save reports
    report_folder = hparams.get_report_dir()
    # evaluation, TODO more reports
    for one_output_hparam in output_hparams:
        ground_truth = one_output_hparam.ground_truth
        predictions = one_output_hparam.predictions
        if one_output_hparam.type in [CLASSLABEL, LIST_OF_CLASSLABEL, LIST_OF_INT]:
            # some filename
            cur_report_folder = os.path.join(report_folder, f'{one_output_hparam.name}_{one_output_hparam.type.lower()}')
            if not os.path.exists(cur_report_folder):
                os.makedirs(cur_report_folder)

            if one_output_hparam.task == NER:
                labels = one_output_hparam.labels
                # confusion matrix
                cm = ConfusionMatrix(_2d_to_1d_list(ground_truth), _2d_to_1d_list(predictions), labels)
                # ner evaluation
                labels = list(set([itm[2:] for itm in labels if itm.startswith("B-") or itm.startswith("I-")]))
                ner_eval = NEREvaluator(
                    _id_to_label(ground_truth, one_output_hparam.labels),
                    _id_to_label(predictions, one_output_hparam.labels),
                    labels)
                ner_results, ner_results_agg = ner_eval.evaluate()
                save_json(os.path.join(cur_report_folder, "ner_results.json"), ner_results)
                save_json(os.path.join(cur_report_folder, "ner_results_agg.json"), ner_results_agg)
            else:
                cm = ConfusionMatrix(ground_truth, predictions, one_output_hparam.labels)

            # print some reports
            print_boxed(f"{one_output_hparam.name} Evaluation")

            cms = cm.confusion_matrix_visual()
            if len(cm.label2idx) < 10:
                print(cms)
                # save reports to files
                with open(os.path.join(cur_report_folder, "confusion_matrix.txt"), 'w') as f:
                    f.write(cms)
            print()
            print(json.dumps(cm.stats(), indent=4))
            save_json(os.path.join(cur_report_folder, "stats.json"), cm.stats())
            save_json(os.path.join(cur_report_folder, 'per_class_stats.json'), cm.per_class_stats())
            # save reports to hparams
            hparams['performance'] = Hparams()
            hparams.performance["stats"] = cm.stats()
            hparams.performance["per_class_stats"] = cm.per_class_stats()
            logger.info(f"Save {one_output_hparam.name} reports in {cur_report_folder}")
        else:
            logger.warning(f"{one_output_hparam.name}'s evaluation has not be implemented.")


def _2d_to_1d_list(seq2d):
    seq1d = list()
    for item in seq2d:
        seq1d.extend(item)
    return seq1d


def _id_to_label(seq, labels):
    if isinstance(seq, int):
        return labels[seq]
    elif isinstance(seq, (list, tuple)):
        return [_id_to_label(id, labels) for id in seq]
    else:
        raise ValueError("type err.")
