#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,4,6,7
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name cmrc2018 \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 2 3 \
    > err.log 2>&1 &


#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name textcnn_for_classification \
#    --schedule train_and_eval \
#    --config_name tnews_k_fold \
#    --config_dir ./configs/glue_zh \
#    --gpus 0 1 \
#    > err.log 2>&1 &