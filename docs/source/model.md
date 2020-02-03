# Model

## BaseModel

The base class ***BaseModel*** inherits the ***tf.keras.Model*** and ***Registry***, which makes subclasses registerable.
It also implements deploy method for helping generate deployment files.

## Custom Models

Take the bert_for_classification model as an example as following:

```python
@BaseModel.register("bert_for_classification")
class BertForSeqClassification(BaseModel):
    ...
```

The registered name of the model BertForSeqClassification is bert_for_classification.
And the implementation of other functions follows ***tf.keras.Model's*** specification.