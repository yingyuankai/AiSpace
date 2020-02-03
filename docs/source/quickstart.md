# Quickstart

## Training

```
python -u aispace/trainer.py \
    --schedule train_and_eval \
    --config_name CONFIG_NAME \
    --config_dir CONFIG_DIR \
    [--experiment_name EXPERIMENT_NAME] \
    [--model_name MODEL_NAME] \
    [--gpus GPUS] 
```

## Training with resumed model

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

## Average checkpoints

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

## Deployment

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
