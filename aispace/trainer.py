# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-05 11:01
# @Author  : yingyuankai@aliyun.com
# @File    : trainer.py

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aispace.utils.hparams import client, Hparams
from aispace.utils.misc import set_random_seed, set_visible_devices
from aispace.utils.builder_utils import build_callbacks, load_dataset, build_model
from aispace.constants import *
from aispace.utils.eval_utils import evaluation


def experiment(hparams: Hparams):
    logger = logging.getLogger(__name__)
    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{id}" for id in hparams.gpus])
    # build dataset
    train_dataset, dev_dataset, dataset_info = load_dataset(hparams, ret_test=False)
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

    # save model
    logger.info(f'Save model in {hparams.get_model_filename()}')
    model.save_weights(hparams.get_model_filename(), save_format="tf")

    # eval on test dataset and make reports
    evaluation(hparams)

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
    from aispace.utils.checkpoint_utils import average_checkpoints
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


if __name__ == '__main__':
    main(sys.argv)
