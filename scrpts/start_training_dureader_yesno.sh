#!/usr/bin/env bash

# Training
#export CUDA_VISIBLE_DEVICES=1,2
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule train_and_eval \
#    --enable_xla False \
#    --config_name dureader_yesno \
#    --config_dir ./configs/qa \
#    --gpus 0 1 \
#    > err.log 2>&1 &

# Build deployment package
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule deploy \
#    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/test_bert_for_classification_dureader__yesno_119_0 \
#    --config_name dureader_yesno \
#    --config_dir ./configs/qa \
#    --gpus 0 1 \
#    > err.log 2>&1 &

# Deploy using bentoml
##DEPLOY_PATH="save/test_bert_for_classification_119_10/deploy/BertTextClassificationService/20200923161737_E41815"
#DEPLOY_PATH="save/test_bert_for_classification_dureader__yesno_119_0/deploy/BertTextClassificationService/20201022095858_ECD2F0"
#DEPLOY_MODE="serve-gunicorn"
##DEPLOY_MODE="serve"
##export BENTOML__APISERVER__DEFAULT_PORT=5000
#TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5000 --debug --enable-microbatch > dureader_yesno_deploy.log 2>&1 &
#echo "Start dureader_yesno service."