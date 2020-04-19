#!/bin/bash
 
#####################################################
# Copyright (c) 2020 Sogou, Inc. All Rights Reserved
#####################################################
# File:    start_deploy_duee.sh
# Author:  root
# Date:    2020/04/17 15:10:21
# Brief:
#####################################################




TRIGGER_DEPLOY_PATH="/search/data1/yyk/workspace/projects/AiSpace/save/trigger_bert_for_ner_119_3/deploy/BertNerService/20200420001443_E4D41E"
ROLE_DEPLOY_PATH="/search/data1/yyk/workspace/projects/AiSpace/save/role_bert_for_role_ner_119_2/deploy/RoleBertNerService/20200420001632_6B9CC7"

export BENTOML__APISERVER__DEFAULT_PORT=5000
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=6 bentoml serve $TRIGGER_DEPLOY_PATH > trigger_extract_deploy.log 2>&1 &
echo "Start trigger extract service."

export BENTOML__APISERVER__DEFAULT_PORT=5001
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=6 bentoml serve $ROLE_DEPLOY_PATH > role_extract_deploy.log 2>&1 &
echo "Start role extract service."














# vim: set expandtab ts=4 sw=4 sts=4 tw=100
