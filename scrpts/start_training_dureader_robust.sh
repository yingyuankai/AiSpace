#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name dureader_robust \
    --config_dir ./configs/qa \
    --gpus 0 1 2 3 \
    > err.log 2>&1 &