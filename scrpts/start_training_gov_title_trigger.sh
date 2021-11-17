#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_ner \
    --schedule train_and_eval \
    --config_name gov_title_trigger \
    --config_dir ./configs/custom \
    --gpus 0 \
    > gov_title_trigger.log 2>&1 &


# Build deployment package
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_ner \
#    --schedule deploy \
#    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/test_bert_for_ner_gov_title__trigger_119_28 \
#    --config_name gov_title_trigger \
#    --config_dir ./configs/custom \
#    --gpus 0 \
#    > err.log 2>&1 &
#
## Deploy using bentoml
#DEPLOY_PATH="/search/odin/yyk/workspace/AiSpace/save/test_bert_for_ner_gov_title__trigger_119_28/deploy/BertNerService/20210111101630_08EEF3"
#DEPLOY_MODE="serve-gunicorn"
##DEPLOY_MODE="serve"
#TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5000 --debug --enable-microbatch --workers 1 > gov_title_trigger_deploy.log 2>&1 &
#echo "Start gov_title_trigger service."