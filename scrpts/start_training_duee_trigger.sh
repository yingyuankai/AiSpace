#!/usr/bin/env bash

export HANLP_GREEDY_GPU=1
export HANLP_HOME="/search/data1/yyk/data/hanlp_data"
export CUDA_VISIBLE_DEVICES=5,0
nohup python -u aispace/trainer.py \
    --experiment_name trigger \
    --model_name bert_for_ner \
    --schedule train_and_eval \
    --config_name DuEE_trigger \
    --config_dir ./configs/2020_LSTC \
    --random_seed 91 \
    --gpus 0 1 \
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
