#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=1
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule train_and_eval \
#    --config_name ewn \
#    --config_dir ./configs/custom \
#    --gpus 0 \
#    > ewn_err.log 2>&1 &

# Build deployment package
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule deploy \
#    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/test_bert_for_classification_entity_with_nationality_119_14 \
#    --config_name ewn \
#    --config_dir ./configs/custom \
#    --gpus 0 1 \
#    > err.log 2>&1 &

# Deploy using bentoml
#DEPLOY_PATH="save/test_bert_for_classification_entity_with_nationality_119_2/deploy/BertTextClassificationService/20201111102110_209866"
DEPLOY_PATH="save/test_bert_for_classification_entity_with_nationality_119_14/deploy/BertTextClassificationService/20201117122539_5AA389"
DEPLOY_MODE="serve-gunicorn"
#DEPLOY_MODE="serve"
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5000 --debug --enable-microbatch > ewn_deploy.log 2>&1 &
echo "Start ewn service."
