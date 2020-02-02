# AiSpace

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/yingyuankai/AiSpace/master/docs/resource/imgs/aispace_logo_name.png" width="400"/>
    <br>
<p>

<p align="center">
    <a href="https://travis-ci.com/yingyuankai/AiSpace.svg?branch=master">
        <img alt="Build" src="https://travis-ci.com/yingyuankai/AiSpace.svg?branch=master">
    </a>
    <a href="https://github.com/yingyuankai/AiSpace/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/yingyuankai/AiSpace.svg?color=blue">
    </a>
    <a href="https://aispace.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://img.shields.io/website/http/aispace.readthedocs.io/en/latest.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/yingyuankai/AiSpace/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/yingyuankai/AiSpace.svg">
    </a>
</p>

AiSpace provides highly configurable framework for deep learning model development, deployment and 
conveniently use of pre-trained models (bert, albert, opt, etc.). 

Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Instructions](#instructions)
  * [Configuration](#Configuration)
  * [Pretrained](#Pretrained)
  * [Experiments](#experiments)
  * [Todos](#Todos)
  * [Refs](#Refs)
  

## Features

* Highly configurable, we manage all hyperparameters with inheritable Configuration files.
* All modules are registerable, including models, dataset, losses, optimizers, metrics, callbacks, etc.
* Standardized process
* Multi-GPU Training
* Integrate multiple pre-trained models, including chinese
* Simple and fast deployment using [BentoML](https://github.com/bentoml/BentoML)
* Integrated Chinese benchmarks [CLUE](https://github.com/CLUEbenchmark/CLUE)

## Requirements

```
git clone https://github.com/yingyuankai/AiSpace.git
cd AiSpace && pip install -r requirements
```

## Instructions

### Training

```
python -u aispace/trainer.py \
    --schedule train_and_eval \
    --config_name CONFIG_NAME \
    --config_dir CONFIG_DIR \
    [--experiment_name EXPERIMENT_NAME] \
    [--model_name MODEL_NAME] \
    [--gpus GPUS] 
```

### Training with resumed model

```
python -u aispace/trainer.py \
    --schedule train_and_eval \
    --config_name CONFIG_NAME \
    --config_dir CONFIG_DIR \
    --model_resume_path MODEL_RESUME_PATH \
    [--experiment_name EXPERIMENT_NAME] \
    [--model_name MODEL_NAME] \
    [--gpus GPUS] 
```

--model_resume_path is a path to initialization model.

### Average checkpoints

```
python -u aispace/trainer.py \
    --schedule avg_checkpoints \
    --config_name CONFIG_NAME \
    --config_dir CONFIG_DIR \
    --prefix_or_checkpoints PREFIX_OR_CHECKPOINGS \
    [--ckpt_weights CKPT_WEIGHTS] \
    [--experiment_name EXPERIMENT_NAME] \
    [--model_name MODEL_NAME] \
    [--gpus GPUS] 
```

--prefix_or_checkpoints is paths to multiple checkpoints separated by comma.

--ckpt_weights is weights same order as the prefix_or_checkpoints.

### Deployment

Generate deployment files before deployment, you need to specify the model path (--model_resume_path) to be deployed like following.

```
python -u aispace/trainer.py \
    --schedule deploy \
    --config_name CONFIG_NAME \
    --config_dir CONFIG_DIR \
    --model_resume_path MODEL_RESUME_PATH \
    [--experiment_name EXPERIMENT_NAME] \
    [--model_name MODEL_NAME] \
    [--gpus GPUS] 
```

We use [BentoML](https://github.com/bentoml/BentoML) as deploy tool, so your must implement the ***deploy*** function in your model class.

## Configuration

All the configurations are in ***configs***, in which ***base*** (./configs/default/base.yml) is the most basic, any configuration downstream includes this configuration directly or indirectly. 
Before you start, it is best to read this configuration carefully.

Your can use ***includes*** field to load other configurations, then the current configuration inherits the configurations and overrides the same configuration fields. Just like class inheritance, a function of the same name in a subclass overrides a function of the parent class.

The syntax is like this:

merge configuration of bert_huggingface into current.
```yaml
includes:
  - "../pretrain/bert_huggingface.yml"     # relative path
```

## Pretrained

We have integrated multiple pre-trained language models and are constantly expanding。

|Model|#Model|#Chinese model|Download manually？|Refs|Status|
|---|---|---|---|---|---|
|bert|13|1|no|[transformers](https://github.com/huggingface/transformers)|Done|
|albert|8|0|no|[transformers](https://github.com/huggingface/transformers)|Done|
|albert_chinese|9|9|yes|[albert_zh](https://github.com/brightmart/albert_zh)|Done|
|bert_wwm|4|4|yes|[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)|Done|
|xlnet|2|0|no|[transformers](https://github.com/huggingface/transformers)|Processing|
|xlnet_chinese|2|2|yes|[transformers](https://github.com/huggingface/transformers)|Processing|
|ernie|4|2|todo|[ERNIE](https://github.com/PaddlePaddle/ERNIE)|Processing|
|NEZHA|-|-|-|[NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model)|Processing|
|TinyBERT|-|-|-|[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model)|Processing|

For those models that need to be downloaded manually, download, unzip them and modify the path in the corresponding configuration.

Some pre-trained models don't have tensorflow versions, I converted them and made them available for download。

|Model|Refs|tf version|
|---|---|---|
|ERNIE_Base_en_stable-2.0.0|[ERNIE](https://github.com/PaddlePaddle/ERNIE)|[baidu yun](https://pan.baidu.com/s/142YNWLrQhO5hxvq2eCxs5g)|
|ERNIE_stable-1.0.1|[ERNIE](https://github.com/PaddlePaddle/ERNIE)|[baidu yun](https://pan.baidu.com/s/15y-1B7rBW7USIvixIRqXrw)|
|ERNIE_1.0_max-len-512|[ERNIE](https://github.com/PaddlePaddle/ERNIE)|[baidu yun](https://pan.baidu.com/s/1B-gehcnVDnNQIAr6m2dPjg)|
|ERNIE_Large_en_stable-2.0.0|[ERNIE](https://github.com/PaddlePaddle/ERNIE)|[baidu yun](https://pan.baidu.com/s/1hUT9X4K7PL5ETysq2mz0IA)|

## Experiments

### glue_zh/tnews

Tnews is a task of Chinese GLUE, which is a short text classification task from ByteDance.

Run Tnews classification
```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name tnews \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 2 3  \
```

Specify different pretrained model, please change ***includes*** and ***pretrained.name*** in config file.


|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|bert-base-chinese-huggingface|65.020|64.987|62.484|63.017|
|albert_base_zh|62.160|62.514|59.267|60.377|
|albert_base_zh_additional_36k_steps|61.760|61.723|58.534|59.273|
|albert_small_zh_google|62.620|63.819|58.992|59.387|
|albert_large_zh|61.830|61.980|59.843|60.200|
|albert_tiny|60.110|57.118|55.559|56.077|
|albert_tiny_489k|61.130|57.875|57.200|57.332|
|albert_tiny_zh_google|60.860|59.500|57.556|57.702|
|albert_xlarge_zh_177k|63.380|63.603|60.168|60.596|
|albert_xlarge_zh_183k|63.210|**67.161**|59.220|59.599|
|chinese_wwm|64.000|62.747|**64.509**|63.042|
|chinese_wwm_ext|65.020|65.048|62.017|62.688|
|chinese_roberta_wwm_ext|64.860|64.819|63.275|**63.591**|
|chinese_roberta_wwm_large_ext|**65.700**|62.342|61.527|61.664|

**NOTE**: The hyper-parameters used here have not been fine-tuned.

## Todos

- More complete and detailed documentation;
- More pretrained models;
- More evaluations of [CLUE](https://github.com/CLUEbenchmark/CLUE);
- More Chinese dataset;
- Support Pytorch;
- Improve the tokenizer to make it more versatile;

## Refs

- [transformers](https://github.com/huggingface/transformers)

- [UER-py](https://github.com/dbiir/UER-py)

- [Huawei pretrained models](https://github.com/huawei-noah/Pretrained-Language-Model)
