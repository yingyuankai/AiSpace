# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-18 13:08
# @Author  : yingyuankai@aliyun.com
# @File    : builder_utils.py

import math
import os
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.download.download_manager import DownloadConfig

from aispace.utils.hparams import Hparams
from aispace.constants import *


logger = logging.getLogger(__name__)


def load_dataset(hparams: Hparams, ret_train=True, ret_dev=True, ret_test=True, ret_info=True):
    from aispace import datasets
    if ret_train:
        train_dataset, dataset_info = build_dataset(hparams, hparams.dataset.source.train, with_info=True)
    if ret_dev:
        dev_dataset, dataset_info = build_dataset(hparams, hparams.dataset.source.validation, with_info=True)
    if ret_test:
        test_dataset, dataset_info = build_dataset(hparams, hparams.dataset.source.test, with_info=True)

    # check the consistence of tokenizer using in building dataset and now using.
    if hparams.get("dataset", {}).get("tokenizer", {}).get("name", "") != "" and \
            (dataset_info.metadata is None or
             hparams.get("dataset", {}).get("tokenizer", {}).get("name", "") != dataset_info.metadata.get("tokenizer", "")):
        raise ValueError(f'The dataset is built using tokenizer {dataset_info.metadata.get("tokenizer", "")}, '
                         f'however, now is using tokenizer '
                         f'{hparams.get("dataset", {}).get("tokenizer", {}).get("name", "")}!')

    # data mapping
    def build_generator(fields):
        input_names = [itm.get('name') for itm in hparams.dataset.inputs]
        output_names = [itm.get('name') for itm in hparams.dataset.outputs]
        inputs, outputs = {}, {}
        for k, v in fields.items():
            if k in input_names:
                inputs[k] = v
            elif k in output_names:
                outputs[k] = v
            else:
                raise ValueError(f"{k} not in inputs or outputs.")
        return inputs, outputs

    # build batch
    if ret_train:
        train_dataset = train_dataset.\
            map(build_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(hparams.training.shuffle_size).\
            repeat(). \
            batch(hparams.training.batch_size). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        logger.info("Train dataset has loaded.")
    if ret_dev:
        dev_dataset = dev_dataset.\
            map(build_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            repeat(). \
            batch(hparams.training.batch_size). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        logger.info("Validation dataset has loaded.")
    if ret_test:
        test_dataset = test_dataset.\
            map(build_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(hparams.training.batch_size). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        logger.info("Test dataset has loaded.")

    # reset some hparams
    if ret_info:
        print(dataset_info)
        training_hparams = hparams.training
        train_data_size = dataset_info.splits.get("train").num_examples
        validation_data_size = dataset_info.splits.get("validation").num_examples
        steps_per_epoch = int(train_data_size / training_hparams.batch_size)
        num_warmup_steps = \
            int(training_hparams.max_epochs * train_data_size * training_hparams.warmup_factor / training_hparams.batch_size)
        num_warmup_steps = min(steps_per_epoch, num_warmup_steps)

        validation_steps = int(
            math.ceil(validation_data_size / training_hparams.batch_size))
        logger.info("Reset some hparams according to dataset_info:")
        if "steps_per_epoch" not in training_hparams or training_hparams.steps_per_epoch <= 0:
            hparams.cascade_set('training.steps_per_epoch', steps_per_epoch)
            logger.info(f"Set training.steps_per_epoch to {steps_per_epoch}")
        else:
            logger.info(f"Get training.steps_per_epoch is {hparams.training.steps_per_epoch}")
        if "validation_steps" not in training_hparams or training_hparams.validation_steps <= 0:
            hparams.cascade_set('training.validation_steps', validation_steps)
            logger.info(f"Set training.validation_steps to {validation_steps}")
        else:
            logger.info(f"Get training.validation_steps is {hparams.training.validation_steps}")
        if "num_warmup_steps" not in training_hparams or training_hparams.num_warmup_steps <= 0:
            hparams.cascade_set('training.num_warmup_steps', num_warmup_steps)
            logger.info(f"Set training.num_warmup_steps to {num_warmup_steps}")
        else:
            logger.info(f"Get training.num_warmup_steps is {hparams.training.num_warmup_steps}")

    result = ()
    if ret_train:
        result += (train_dataset, )
    if ret_dev:
        result += (dev_dataset, )
    if ret_test:
        result += (test_dataset, )

    if ret_info:
        result += (dataset_info, )

    return result


def build_dataset(hparams: Hparams, split=None, with_info=False):
    checksum_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/url_checksums")
    tfds.download.add_checksums_dir(checksum_dir)
    download_mode = tfds.core.download.GenerateMode.FORCE_REDOWNLOAD \
        if hparams.force_rebuild_data and split == hparams.dataset.source.train else tfds.core.download.GenerateMode.REUSE_DATASET_IF_EXISTS
    download_config = DownloadConfig(download_mode=download_mode)
    return tfds.load(hparams.dataset.name, split=split,
                     with_info=with_info,
                     # as_supervised=as_supervised,
                     data_dir=hparams.dataset.data_dir,
                     builder_kwargs={'hparams': hparams},
                     download_and_prepare_kwargs={'download_config': download_config}
                     )


def build_callbacks(hparams: Hparams):
    from aispace.layers.callbacks import CALLBACKS
    callbacks = []
    for name, config in hparams.items():
        if not config.switch: continue
        fn = CALLBACKS.get(name)
        logger.info(f"Using callback [{name}].")
        if fn is None:
            logger.warning(f"Callback name {name} may be wrong.")
            continue
        callbacks.append(fn(**config.config))
    return callbacks


def build_model(hparam: Hparams, return_losses=True, return_metrics=True, return_optimizer=True):
    """Build custom keras model, losses, metrics, and optimizer

    :param hparam:
    :param return_losses:
    :param return_metrics:
    :param return_optimizer:
    :return:
    """
    logger.info(f"Try to build model {hparam.model_name}")
    from aispace import models
    from aispace.models.base_model import BaseModel
    model = BaseModel.by_name(hparam.model_name)(hparam)
    # build inputs and model
    inputs = build_tf_model_inputs(hparam.dataset)
    model(inputs)

    rets = ()
    # build losses
    if return_losses:
        losses, loss_weights = build_tf_model_losses(model, hparam.dataset)
        rets += ((losses, loss_weights),)
    # build metrics
    if return_metrics:
        metrics = build_tf_model_metrics(hparam.dataset)
        rets += (metrics,)
    # build optimizer
    if return_optimizer:
        optimizer = build_tf_model_optimizer(hparam.training)
        rets += (optimizer, )
    model.summary()
    # init from pretrained model (language or etc.,)
    if not hparam.model_resume_path and not hparam.model_load_path \
            and "pretrained" in hparam and hparam.pretrained.init_from_pretrained:
        try:
            logger.info(f"Load weights from {hparam.pretrained.model_path}")
            if hparam.pretrained.model_path.endswith(".h5"):
                model.load_weights(hparam.pretrained.model_path, by_name=True)
            else:
                logger.info(f"Load weights using model adapter {hparam.pretrained.adapter}")
                adapter = build_model_adapter(hparam.pretrained)
                if adapter is not None:
                    adapter(model.trainable_variables, hparam.pretrained.model_path)
        except Exception as e:
            logging.error("Load weights failure!", exc_info=True)
            raise e

    # initializer model
    if not hparam.model_resume_path and hparam.model_load_path is not None:
        model_saved = os.path.join(hparam.model_load_path, "model_saved", "model")
        logger.info(f"Initialize model from {model_saved}")
        model.load_weights(model_saved)

    # resume model
    if hparam.model_resume_path is not None:
        model_saved = os.path.join(hparam.get_workspace_dir(), "model_saved", "model")
        logger.info(f"Resume model from {model_saved}")
        model.load_weights(model_saved)

    return (model,) + rets


# TODO more general
def build_tf_model_inputs(dataset_hparams: Hparams):
    """build input for keras model

    :param input_hparams:
    :return:
    """

    inputs = {}
    for item in dataset_hparams.inputs:
        if item.type == LIST_OF_INT:
            input = tf.keras.layers.Input(
                shape=(item.max_len,), dtype=tf.int32, name=item.name
            )
            inputs[item.name] = input
    return inputs


def build_tf_model_losses(model, dataset_hparams: Hparams):
    """ build losses for outputs, every outputs may be have a loss function.

    :param dataset_hparams:
    :return:
    """
    from aispace.layers.losses import LOSSES
    losses = {}
    loss_weights = {}
    for output_hparam in dataset_hparams.outputs:
        name = output_hparam.name
        loss_weight = output_hparam.weight if "weight" in output_hparam else 1.
        loss_weights[name] = loss_weight
        loss_hparam = output_hparam.loss
        loss_name = loss_hparam.name
        loss_fn = LOSSES.get(loss_name)
        loss_config = loss_hparam.get('config', {})
        # get loss from model
        # loss_name's format likes myself_{loss_property_name}
        if loss_name.startswith(MYSELF_LOSS_PREFIX):
            loss_fn = getattr(model, loss_name[loss_name.find(MYSELF_LOSS_PREFIX) + 1 + len(MYSELF_LOSS_PREFIX):])
        logger.info(f"Using loss [{loss_hparam.name}] for {name}.")
        if loss_fn is None:
            logger.warning(f"{name}'s loss [{loss_hparam.name}] does not registered.")
        losses[name] = loss_fn(loss_config)
    assert any([v is not None for k, v in losses.items()]), \
        "Can not get losses from configuration."
    return losses, loss_weights


def build_tf_model_metrics(dataset_hparams: Hparams):
    """build metrics for outputs, every outputs may be have multi-metrics.

    :param dataset_hparams:
    :return:
    """
    from aispace.layers.metrics import METRICS
    res_metrics = {}
    for output_hparam in dataset_hparams.outputs:
        name = output_hparam.name
        metrics = output_hparam.metrics
        if name not in metrics:
            res_metrics[name] = []
        for idx, metric in enumerate(metrics):
            cur_config = metric.get("config", {})
            metric_fn = METRICS.get(metric.name)(cur_config)
            logger.info(f"Using metric[{idx}] [{metric.name}] for {name}.")
            if metric_fn is None:
                logger.warning(f"metric [{metric.name}] does not registered.")
            res_metrics[name].append(metric_fn)
    return res_metrics


def build_tf_model_optimizer(train_hparams: Hparams):
    """build optimizer

    :param train_hparams:
    :return:
    """
    from aispace.layers.optimizers import OPTIMIZERS
    optimizer_fn = OPTIMIZERS.get(train_hparams.optimizer.name)
    logger.info(f"Using optimizer [{train_hparams.optimizer.name}].")
    if optimizer_fn is None:
        logger.warning(f"Optimizer [{train_hparams.optimizer.name}] does not registered.")
        logger.warning(f"Change to default optimizer adma.")
        optimizer_fn = OPTIMIZERS.get('adam')
    optimizer = optimizer_fn(train_hparams)
    return optimizer


def build_model_adapter(pretrained_hparams: Hparams):
    from aispace.layers.adapters import ADAPTERS
    adapter_name = pretrained_hparams.adapter
    if adapter_name is None:
        logger.error("Do not specify any adapter.")
        return None
    adapter_fn = ADAPTERS.get(adapter_name)
    if adapter_fn is None:
        logger.error(f"Adapter [{adapter_name}] does not registered.")
        return None
    return adapter_fn