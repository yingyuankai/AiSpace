#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3,4,5,6
nohup python -u aispace/trainer.py \
    --experiment_name trigger \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name DuEE_trigger_as_classifier \
    --config_dir ./configs/2020_LSTC \
    --gpus 0 1 2 3 \
    > err.log 2>&1 &


#export CUDA_VISIBLE_DEVICES=7
#nohup python -u aispace/trainer.py \
#    --experiment_name trigger \
#    --model_name bert_for_ner \
#    --model_resume_path /search/data1/yyk/workspace/projects/AiSpace/save/trigger_bert_for_ner_119_20 \
#    --schedule deploy \
#    --config_name DuEE_trigger \
#    --config_dir ./configs/2020_LSTC \
#    --gpus 0 \
#    > trigger_deploy_err.log 2>&1 &
