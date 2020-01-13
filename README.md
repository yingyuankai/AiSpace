# AiSpace

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/yingyuankai/AiSpace/master/docs/resource/imgs/aispace_logo_name.png" width="400"/>
    <br>
<p>



Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Instructions](#instructions)
  * [Experiments](#experiments)
  

## Features

* Highly configurable
* Standardized process
* Multi-GPU Training
* Integrate multiple pre-trained models, including Chinese
* Simple and fast deployment
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
