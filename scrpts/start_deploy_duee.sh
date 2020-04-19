#!/bin/bash
 
#####################################################
# Copyright (c) 2020 Sogou, Inc. All Rights Reserved
#####################################################
# File:    start_deploy_duee.sh
# Author:  root
# Date:    2020/04/17 15:10:21
# Brief:
#####################################################




TRIGGER_DEPLOY_PATH="/search/data1/yyk/workspace/projects/AiSpace/save/trigger_bert_for_ner_119_0/deploy/BertNerService/20200417164029_5894BC"
ROLE_DEPLOY_PATH="/search/data1/yyk/workspace/projects/AiSpace/save/role_bert_for_role_ner_119_0/deploy/RoleBertNerService/20200419173622_9ED071"

export BENTOML__APISERVER__DEFAULT_PORT=5000
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=6 bentoml serve $TRIGGER_DEPLOY_PATH > trigger_extract_deploy.log 2>&1 &
echo "Start trigger extract service."

export BENTOML__APISERVER__DEFAULT_PORT=5001
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=6 bentoml serve $ROLE_DEPLOY_PATH > role_extract_deploy.log 2>&1 &
echo "Start role extract service."














# vim: set expandtab ts=4 sw=4 sts=4 tw=100
