#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name dureader_yesno \
    --config_dir ./configs/qa \
    --gpus 0 1 \
    > err.log 2>&1 &