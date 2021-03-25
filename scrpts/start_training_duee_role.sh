#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,0
nohup python -u aispace/trainer.py \
    --experiment_name role \
    --model_name bert_for_role_ner \
    --schedule train_and_eval \
    --config_name DuEE_role \
    --config_dir ./configs/2020_LSTC \
    --gpus 0 1 \
    > duee_role.log 2>&1 &


#export CUDA_VISIBLE_DEVICES=7
#nohup python -u aispace/trainer.py \
#    --experiment_name role \
#    --model_name bert_for_role_ner \
#    --model_resume_path /search/data1/yyk/workspace/projects/AiSpace/save/role_bert_for_role_ner_119_2 \
#    --schedule deploy \
#    --config_name DuEE_role \
#    --config_dir ./configs/2020_LSTC \
#    --gpus 0 \
#    > role_deploy_err.log 2>&1 &