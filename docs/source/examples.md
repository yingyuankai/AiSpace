# Examples

## glue_zh/tnews

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
|albert_base_zh_google|59.340|59.787|55.659|56.788|
|albert_large_zh_google|58.210|59.465|54.548|55.308|
|chinese_wwm|64.000|62.747|**64.509**|63.042|
|chinese_wwm_ext|65.020|65.048|62.017|62.688|
|chinese_roberta_wwm_ext|64.860|64.819|63.275|**63.591**|
|chinese_roberta_wwm_large_ext|**65.700**|62.342|61.527|61.664|
|ERNIE_stable-1.0.1|**66.330**|66.903|63.704|64.524|
|ERNIE_1.0_max-len-512|66.010|65.301|62.230|62.884|
|chinese_xlnet_base|65.110|64.377|**64.862**|64.169|
|chinese_xlnet_mid|66.000|66.377|63.874|**64.708**|
|chinese_electra_small|60.370|60.223|57.161|57.206|
|chinese_electra_small_ex|59.900|58.078|55.525|56.194|
|chinese_electra_base|60.500|60.090|58.267|58.909|
|chinese_electra_large|60.500|60.362|57.653|58.336|
|nezha-base|58.940|57.909|55.650|55.630|
|nezha-base-wwm|58.800|60.060|54.859|55.831|

**NOTE**: The hyper-parameters used here have not been fine-tuned.

## glue_zh/cmrc2018

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name cmrc2018 \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 2 3 \
    > err.log 2>&1 &
```

|Model|F1|EM|
|---|---|---|
|bert-base-chinese-huggingface|71.718|44.419|
|albert_base_zh|69.463|41.643|
|albert_base_zh_google|68.538|39.320|
|chinese_wwm|72.081|44.419|
|chinese_roberta_wwm_ext|71.523|44.362|
|ERNIE_stable-1.0.1|**83.835**|64.898|
|ERNIE_1.0_max-len-512|83.363|**65.293**|
|chinese_electra_small|72.172|46.314|

## glue_zh/csl

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name csl \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > csl_err.log 2>&1 &
```

|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|ERNIE_1.0_max-len-512|83.000|83.439|83.000|82.943|

## glue_zh/drcd

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name drcd \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > drcd_err.log 2>&1 &
```

|Model|F1|EM|
|---|---|---|
|ERNIE_1.0_max-len-512|85.657|75.433|

## glue_zh/afqmc

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name afqmc \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > afqmc_err.log 2>&1 &
```

|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|ERNIE_1.0_max-len-512|72.405|67.489|66.750|67.071|

## glue_zh/iflytek

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name iflytek \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > iflytek_err.log 2>&1 &
```

|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|ERNIE_1.0_max-len-512|58.753|30.406|32.275|28.965|

## glue_zh/cmnli

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --config_name cmnli \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > cmnli_err.log 2>&1 &
```

|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|ERNIE_1.0_max-len-512|78.759|78.750|78.679|78.593|

## glue_zh/wsc

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_relation_extract \
    --schedule train_and_eval \
    --config_name wsc \
    --config_dir ./configs/glue_zh \
    --gpus 0 1 \
    > wsc_err.log 2>&1 &
```

|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|ERNIE_1.0_max-len-512|59.615|58.507|55.969|54.117|

## dureader/robust

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_qa \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name dureader_robust \
    --config_dir ./configs/qa \
    --gpus 0 1 \
    > err.log 2>&1 &
```
|Model|F1|EM|
|---|---|---|
|bert-base-chinese-huggingface|66.624|51.856|
|chinese_wwm|67.007|53.434|
|chinese_roberta_wwm_ext|65.521|50.274|
|ERNIE_stable-1.0.1|75.268|61.675|
|ERNIE_1.0_max-len-512|**83.609**|**72.328**|

## dureader/yesno

```
python -u aispace/trainer.py \
    --experiment_name test \
    --model_name bert_for_classification \
    --schedule train_and_eval \
    --enable_xla False \
    --config_name dureader_yesno \
    --config_dir ./configs/qa \
    --gpus 0 1 \
    > err.log 2>&1 &
```

|Model|Accuracy|Macro_precision|Macro_recall|Macro_f1|
|---|---|---|---|---|
|bert-base-chinese-huggingface|76.565|73.315|69.958|71.230|
|ERNIE_stable-1.0.1|85.756|82.919|81.627|82.213|
|ERNIE_1.0_max-len-512|86.122|83.847|80.636|81.965|