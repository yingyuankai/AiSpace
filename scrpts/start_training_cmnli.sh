#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name cmnli \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > cmnli_err.log 2>&1 &