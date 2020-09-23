#!/usr/bin/env bash

# training
export CUDA_VISIBLE_DEVICES=1,2
nohup python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name dureader_robust \
    --config_dir ./configs/qa \
    --gpus 0 1 \
    > err.log 2>&1 &

# Build deployment package
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_qa \
#    --schedule deploy \
#    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/test_bert_for_qa_119_87 \
#    --config_name dureader_robust \
#    --config_dir ./configs/qa \
#    --gpus 0 1 \
#    > err.log 2>&1 &

## Deploy using bentoml
#DEPLOY_PATH="/search/odin/yyk/workspace/AiSpace/save/test_bert_for_qa_119_87/deploy/BertQAService/20200923203310_DA2F44"
#DEPLOY_MODE="serve-gunicorn"
##DEPLOY_MODE="serve"
#TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5001 --debug --enable-microbatch --workers 1 > dureader_robust_deploy.log 2>&1 &
#echo "Start dureader_robust service."