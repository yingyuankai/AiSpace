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

## Output file structure

The default output path is ***save***, which may has multiple output directories under name as:

```text
{experiment_name}_{model_name}_{random_seed}_{id}
```

Where ***id*** indicates the sequence number of the experiment for the same task, increasing from 0.

Take the text classification task as an example, the output file structure is similar to the following:

```
test_bert_for_classification_119_0
├── checkpoint                  # 1. checkpoints
│   ├── checkpoint
│   ├── ckpt_1.data-00000-of-00002
│   ├── ckpt_1.data-00001-of-00002
│   ├── ckpt_1.index
|   ...
├── deploy                      # 2. Bentoml depolyment directory
│   └── BertTextClassificationService
│       └── 20191208180211_B6FC81
├── hparams.json                # 3. Json file of all hyperparameters
├── logs                        # 4. general or tensorboard log directory
│   ├── errors.log              # error log file
│   ├── info.log                # info log file
│   ├── train                
│   │   ├── events.out.tfevents.1574839601.jshd-60-31.179552.14276.v2
│   │   ├── events.out.tfevents.1574839753.jshd-60-31.profile-empty
│   └── validation
│       └── events.out.tfevents.1574839787.jshd-60-31.179552.151385.v2
├── model_saved                 # 5. last model saved
│   ├── checkpoint
│   ├── model.data-00000-of-00002
│   ├── model.data-00001-of-00002
│   └── model.index
└── reports                     # 6. Eval reports for every output or task
    └── output_1_classlabel     # For example, text classification task
        ├── confusion_matrix.txt
        ├── per_class_stats.json
        └── stats.json
```