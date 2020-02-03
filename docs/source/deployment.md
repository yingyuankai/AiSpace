# Deployment

We use [BentoML](https://github.com/bentoml/BentoML) as deploy tool, so your must implement the ***deploy*** function in your model class and a **bentoml service class**.

Take model ***bert_for_classification*** as an example:

## Custom bentoml service

For more detailed information, please visit [BentoML](https://github.com/bentoml/BentoML).

```python
__all__ = [
    "BertTextClassificationService"
]

import os, sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../" * 4)))

from bentoml import api, env, BentoService, artifacts
from bentoml.artifact import TensorflowSavedModelArtifact, PickleArtifact
from bentoml.handlers import JsonHandler

import numpy as np
from scipy.special import softmax

from aispace.datasets.tokenizer import BertTokenizer
from aispace.utils.hparams import Hparams

@artifacts([
        TensorflowSavedModelArtifact('model'),
        PickleArtifact('tokenizer'),
        PickleArtifact("hparams"),
    ])
@env(pip_dependencies=['tensorflow-gpu==2.0.0', 'numpy==1.16', 'scipy==1.3.1', "tensorflow-datasets==1.3.0"])
class BertTextClassificationService(BentoService):

    def preprocessing(self, text_str):
        input_ids, token_type_ids, attention_mask = self.artifacts.tokenizer.encode(text_str)
        return input_ids, token_type_ids, attention_mask

    def decode_label_idx(self, idx):
        return self.artifacts.hparams.dataset.outputs[0].labels[idx]

    @api(JsonHandler)
    def title_predict(self, parsed_json):
        input_data = {
            "input_ids": [], "token_type_ids": [], "attention_mask": []
        }
        if isinstance(parsed_json, (list, tuple)):
            pre_input_data = list(zip(*list(map(self.preprocessing, parsed_json))))
            input_data['input_ids'].extend(pre_input_data[0])
            input_data['token_type_ids'].extend(pre_input_data[1])
            input_data['attention_mask'].extend(pre_input_data[2])
        else:  # expecting type(parsed_json) == dict:
            pre_input_data = self.preprocessing(parsed_json['text'])
            input_data['input_ids'].append(pre_input_data[0])
            input_data['token_type_ids'].append(pre_input_data[1])
            input_data['attention_mask'].append(pre_input_data[2])

        input_data['input_ids'] = tf.constant(input_data['input_ids'], name="input_ids")
        input_data['token_type_ids'] = tf.constant(input_data['token_type_ids'], name="token_type_ids")
        input_data['attention_mask'] = tf.constant(input_data['attention_mask'], name="attention_mask")
        prediction = self.artifacts.model(input_data, training=False)
        prediction_normed = softmax(prediction[0].numpy(), -1)
        prediction_idx = np.argmax(prediction_normed, -1).tolist()
        prediction_confidence = np.max(prediction_normed, -1).tolist()
        ret = {
            "predictions": []
        }
        for idx, confidence in zip(prediction_idx, prediction_confidence):
            cur_label = self.decode_label_idx(idx)
            new_ret = {
                "label": cur_label,
                "confidence": confidence
            }
            ret["predictions"].append(new_ret)

        return ret
```

## Deploy function
Deploy function in model class as following

```python
def deploy(self):
    """Return path of deployment files"""
    from aispace.datasets.tokenizer import BertTokenizer
    from .bento_services import BertTextClassificationService
    tokenizer = BertTokenizer(self._hparams.dataset.tokenizer)
    bento_service = \
        BertTextClassificationService.pack(
            model=self,
            tokenizer=tokenizer,
            hparams=self._hparams,
        )
    saved_path = bento_service.save(self._hparams.get_deploy_dir())
    return saved_path
```
## Generate deployment files

To generate deployment files, you need to specify the model path (--model_resume_path) to be deployed and run following script.

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

