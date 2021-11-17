#!/usr/bin/env bash

#export HANLP_GREEDY_GPU=1
#export HANLP_HOME="/search/data1/yyk/data/hanlp_data"
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
export CUDA_VISIBLE_DEVICES=0,1
nohup python -u aispace/trainer.py \
    --experiment_name trigger \
    --model_name bert_dgcnn_for_ner \
    --schedule train_and_eval \
    --config_name DuEE_trigger \
    --config_dir ./configs/2020_LSTC \
    --random_seed 91 \
    --use_mixed_float16 True \
    --gpus 0 1 \
    > duee_trigger.log 2>&1 &


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
