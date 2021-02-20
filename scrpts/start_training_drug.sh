#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_ner \
#    --schedule train_and_eval \
#    --config_name drug \
#    --config_dir ./configs/custom \
#    --gpus 0 \
#    > drug.log 2>&1 &


# Build deployment package
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_ner \
#    --schedule deploy \
#    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/test_bert_for_ner_drug__drug_119_2 \
#    --config_name drug \
#    --config_dir ./configs/custom \
#    --gpus 0 \
#    > err.log 2>&1 &
#
# Deploy using bentoml
DEPLOY_PATH="/search/odin/yyk/workspace/AiSpace/save/test_bert_for_ner_drug__drug_119_2/deploy/BertNerService/20210220150302_BE8F8F"
DEPLOY_MODE="serve-gunicorn"
#DEPLOY_MODE="serve"
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5002 --debug --enable-microbatch --workers 1 > drug_deploy.log 2>&1 &
echo "Start drug service."