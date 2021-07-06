#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=1,0
#nohup python -u aispace/trainer.py \
#    --experiment_name keyphrase \
#    --model_name bert_for_ner \
#    --schedule train_and_eval \
#    --config_name DuEE_keyphrase \
#    --config_dir ./configs/2020_LSTC \
#    --gpus 0 1 \
#    > duee_keyphrase.log 2>&1 &


# Build deployment package
#nohup python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_ner \
#    --schedule deploy \
#    --model_resume_path /search/odin/yyk/workspace/AiSpace/save/keyphrase_bert_for_ner_lstc_2020__DuEE_role_119_13 \
#    --config_name DuEE_role \
#    --config_dir ./configs/2020_LSTC \
#    --gpus 0 \
#    > err.log 2>&1 &

# Deploy using bentoml
DEPLOY_PATH=" /search/odin/yyk/workspace/AiSpace/save/keyphrase_bert_for_ner_lstc_2020__DuEE_role_119_13/deploy/BertNerService/20210701171400_6DB6C2"
DEPLOY_MODE="serve-gunicorn"
#DEPLOY_MODE="serve"
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 bentoml ${DEPLOY_MODE} ${DEPLOY_PATH} --port 5004 --debug --enable-microbatch --workers 1 > event_keyphrase_deploy.log 2>&1 &
echo "Start event_keyphrase service."