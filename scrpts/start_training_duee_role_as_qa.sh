#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=1,0,2,3
#nohup python -u aispace/trainer.py \
#    --experiment_name role \
#    --model_name bert_for_qa \
#    --schedule train_and_eval \
#    --model_load_path save/test_bert_for_qa_glue_zh__cmrc2018_119_8 \
#    --config_name DuEE_role_as_qa \
#    --config_dir ./configs/2020_LSTC \
#    --gpus 0 1 2 3 \
#    > duee_role.log 2>&1 &


# Build deployment package
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule deploy \
    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/role_bert_for_qa_lstc_2020__DuEE_role_119_17 \
    --config_name DuEE_role_as_qa \
    --config_dir ./configs/2020_LSTC \
    --gpus 0 \
    > err.log 2>&1 &

# Deploy using bentoml
#DEPLOY_PATH=" /search/odin/yyk/workspace/AiSpace/save/role_bert_for_qa_lstc_2020__DuEE_role_119_3/deploy/BertQAService/20210420141619_E173B6"
#DEPLOY_PATH=" /search/odin/yyk/workspace/AiSpace/save/role_bert_for_qa_lstc_2020__DuEE_role_119_9/deploy/BertQAService/20210422171656_501205"
DEPLOY_PATH=" /search/odin/yyk/workspace/AiSpace/save/role_bert_for_qa_lstc_2020__DuEE_role_119_17/deploy/BertQAService//20210423143146_60581D"
DEPLOY_MODE="serve-gunicorn"
#DEPLOY_MODE="serve"
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5003 --debug --enable-microbatch --workers 1 > event_role_as_qa_deploy.log 2>&1 &
echo "Start dureader_robust service."
