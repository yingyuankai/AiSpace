#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python -u aispace/trainer.py \
    --experiment_name trigger \
    --model_name bert_for_ner \
    --schedule train_and_eval \
    --config_name DuEE_trigger \
    --config_dir ./configs/2020_LSTC \
    --gpus 0 1 2 3 \
    > err.log 2>&1 &