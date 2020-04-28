#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=5,6
#nohup python -u aispace/trainer.py \
#    --experiment_name role \
#    --model_name bert_for_role_ner_v2 \
#    --schedule train_and_eval \
#    --config_name DuEE_role_reduce_label \
#    --config_dir ./configs/2020_LSTC \
#    --gpus 0 1 \
#    > err.log 2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup python -u aispace/trainer.py \
    --experiment_name role \
    --model_name bert_for_role_ner_v2 \
    --model_resume_path /search/data1/yyk/workspace/projects/AiSpace/save/role_bert_for_role_ner_v2_119_70 \
    --schedule deploy \
    --config_name DuEE_role_reduce_label \
    --config_dir ./configs/2020_LSTC \
    --gpus 0 \
    > role_deploy_err.log 2>&1 &