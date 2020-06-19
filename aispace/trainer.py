# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-05 11:01
# @Author  : yingyuankai@aliyun.com
# @File    : trainer.py

import os
import sys
import logging
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aispace.utils.hparams import client, Hparams
from aispace.utils.misc import set_random_seed, set_visible_devices
from aispace.utils.builder_utils import build_callbacks, load_dataset, build_model
from aispace.constants import *
from aispace.utils.eval_utils import evaluation
from aispace.utils.checkpoint_utils import average_checkpoints


def experiment(hparams: Hparams):
    logger = logging.getLogger(__name__)
    if hparams.use_mixed_float16:
        logger.info("Use auto mixed policy")
        # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{id}" for id in hparams.gpus])
    # build dataset
    train_dataset, dev_dataset, dataset_info = next(load_dataset(hparams, ret_test=False))

    with strategy.scope():
        # build model
        model, (losses, loss_weights), metrics, optimizer = build_model(hparams)
        # build callbacks
        callbacks = build_callbacks(hparams.training.callbacks)
        # compile
        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )
        # fit
        model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=hparams.training.max_epochs,
            callbacks=callbacks,
            steps_per_epoch=hparams.training.steps_per_epoch,
            validation_steps=hparams.training.validation_steps,
        )

    # 进行lr finder
    lr_finder_call_back = [cb for cb in callbacks if hasattr(cb, "lr_finder_plot")]
    if len(lr_finder_call_back) != 0:
        logger.info(f"Do lr finder, and save result in {hparams.get_lr_finder_jpg_file()}")
        lr_finder_call_back[0].lr_finder_plot(hparams.get_lr_finder_jpg_file())
    else:
        # load best model
        checkpoint_dir = os.path.join(hparams.get_workspace_dir(), "checkpoint")
        if hparams.eval_use_best and os.path.exists(checkpoint_dir):
            logger.info(f"Load best model from {checkpoint_dir}")
            average_checkpoints(model, checkpoint_dir)
        # save best model
        logger.info(f'Save model in {hparams.get_model_filename()}')
        model.save_weights(hparams.get_model_filename(), save_format="tf")

        # eval on test dataset and make reports
        evaluation(hparams)

    logger.info('Experiment Finish!')


def k_fold_experiment(hparams: Hparams):
    """
    k_fold training
    :param hparams:
    :return:
    """
    logger = logging.getLogger(__name__)
    if hparams.use_mixed_float16:
        logger.info("Use auto mixed policy")
        # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{id}" for id in hparams.gpus])
    # build dataset

    model_saved_dirs = []

    for idx, (train_dataset, dev_dataset, dataset_info) in enumerate(load_dataset(hparams, ret_test=False)):
        logger.info(f"Start {idx}th-fold training")
        with strategy.scope():
            # build model
            model, (losses, loss_weights), metrics, optimizer = build_model(hparams)
            # build callbacks
            callbacks = build_callbacks(hparams.training.callbacks)
            # compile
            model.compile(
                optimizer=optimizer,
                loss=losses,
                metrics=metrics,
                loss_weights=loss_weights
            )
            # fit
            model.fit(
                train_dataset,
                validation_data=dev_dataset,
                epochs=hparams.training.max_epochs,
                callbacks=callbacks,
                steps_per_epoch=hparams.training.steps_per_epoch,
                validation_steps=hparams.training.validation_steps,
            )

            # build archive dir
            k_fold_dir = os.path.join(hparams.get_workspace_dir(), "k_fold", str(idx))
            if not os.path.exists(k_fold_dir):
                os.makedirs(k_fold_dir)

            # load best model
            checkpoint_dir = os.path.join(hparams.get_workspace_dir(), "checkpoint")
            if hparams.eval_use_best and os.path.exists(checkpoint_dir):
                logger.info(f"Load best model from {checkpoint_dir}")
                average_checkpoints(model, checkpoint_dir)
                logger.info(f"Move {checkpoint_dir, k_fold_dir}")
                shutil.move(checkpoint_dir, k_fold_dir)

            # save best model
            logger.info(f'Save {idx}th model in {hparams.get_model_filename()}')
            model.save_weights(hparams.get_model_filename(), save_format="tf")

        # eval on test dataset and make reports
        evaluation(hparams)
        logger.info(f"Move {hparams.get_report_dir()} to {k_fold_dir}")
        shutil.move(hparams.get_report_dir(), k_fold_dir)
        logger.info(f"Move {hparams.get_saved_model_dir()} to {k_fold_dir}")
        cur_model_saved_dir = shutil.move(hparams.get_saved_model_dir(), k_fold_dir)
        logger.info(f"New model saved path for {idx}th fold: {cur_model_saved_dir}")
        model_saved_dirs.append(cur_model_saved_dir)

        logger.info(f'{idx}th-fold experiment Finish!')

    # eval on test dataset after average_checkpoints
    # logger.info("Average models of all fold models.")
    checkpoints = [f'{itm}/model' for itm in model_saved_dirs]
    # average_checkpoints(model, checkpoints)

    # logger.info(f"Save averaged model in {hparams.get_model_filename()}")
    # model.save_weights(hparams.get_model_filename(), save_format="tf")

    evaluation(hparams, checkpoints=checkpoints)

    logger.info('Experiment Finish!')


def deploy(hparams: Hparams):
    logger = logging.getLogger(__name__)
    assert hparams.model_resume_path is not None, ValueError("Model resume path is None, must be specified.")
    # reuse hparams
    model_resume_path = hparams.model_resume_path
    logger.info(f"Reuse saved json config from {os.path.join(hparams.get_workspace_dir(), 'hparams.json')}")
    hparams.reuse_saved_json_hparam()
    hparams.cascade_set("model_resume_path", model_resume_path)
    # build model
    (model,) = build_model(hparams, return_losses=False, return_metrics=False, return_optimizer=False)
    logger.info("Export model to deployment.")
    saved_path = model.deploy()
    logger.info(f"Save bento Service in {saved_path}")


def avg_checkpints(hparams: Hparams):
    logger = logging.getLogger(__name__)
    (model,) = build_model(hparams, return_losses=False, return_metrics=False, return_optimizer=False)
    logger.info(f"Average checkpoints from {hparams.prefix_or_checkpints}")
    average_checkpoints(model, hparams.prefix_or_checkpints, hparams.num_last_checkpoints, hparams.ckpt_weights)
    evaluation(hparams, model=model)
    logger.info(f"Save model in {hparams.get_model_filename()}")
    model.save_weights(hparams.get_model_filename(), save_format="tf")


def main(argv):
    """ Entry Point """
    # build hparams from sys argv or yaml file
    hparams: Hparams = client(argv[1:])
    # prepare resource
    hparams.stand_by()
    # set GPU
    set_visible_devices(hparams.gpus)
    # set random seed for tensorflow, numpy, and random
    set_random_seed(hparams.random_seed)
    # print logo and version
    print(LOGO_STR)
    logging.info(f"Workspace: {hparams.get_workspace_dir()}")
    # experiment
    try:
        if hparams.schedule in ["train_and_eval"]:
            if hparams.training.policy.name == "k-fold":
                k_fold_experiment(hparams)
            else:
                experiment(hparams)
            hparams.to_json()
            logging.info(f"save hparams to {hparams.hparams_json_file}")
        elif hparams.schedule == "eval":
            logger = logging.getLogger(__name__)
            assert hparams.model_resume_path is not None, ValueError("Model resume path is None, must be specified.")
            logger.info(f"Reuse saved json config from {os.path.join(hparams.get_workspace_dir(), 'hparams.json')}")
            hparams.reuse_saved_json_hparam()
            evaluation(hparams)
        elif hparams.schedule == "deploy":
            deploy(hparams)
        elif hparams.schedule == "avg_checkpoints":
            avg_checkpints(hparams)
        else:
            raise NotImplementedError
    except Exception as e:
        logging.error("Error!", exc_info=True)
        hparams.to_json()
        logging.info(f"save hparams to {hparams.hparams_json_file}")


if __name__ == '__main__':
    main(sys.argv)
