#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name ewn \
    --config_dir ./configs/custom \
    --gpus 0 1 \
    > ewn_err.log 2>&1 &
