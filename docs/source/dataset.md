# Dataset

## BaseDataset

The base class ***BaseDataset*** inherits the ***tfds.core.GeneratorBasedBuilder*** and ***Registry***, which makes subclasses registerable.

## Custom dataset

Take the glue_zh dataset as an example as following:

```python
@BaseDataset.register("glue_zh")
class GlueZh(BaseDataset):
    ...
```
The development follows ***[tensorflow_dataset's](https://www.tensorflow.org/datasets)*** specification.

