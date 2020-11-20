# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:24
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : entity_with_nationality.py


__all__ = ["EntityWithNationality"]

import os
import six
import logging
import json
import tensorflow as tf
import tensorflow_datasets as tfds

from aispace.datasets import BaseDataset
from aispace.datasets import data_transformers as Transformer

_GLUE_CITATION = "TODO"

logger = logging.getLogger(__name__)


class EntityWithNationalityConfig(tfds.core.BuilderConfig):
    """BuilderConfig for GLUE."""

    # @tfds.core.disallow_positional_args
    def __init__(self,
                 text_features=None,
                 label_column=None,
                 data_url=None,
                 data_dir=None,
                 citation=None,
                 url=None,
                 label_classes=None,
                 train_shards=1,
                 process_label=lambda x: x,
                 **kwargs):
        """BuilderConfig for GLUE.

    Args:
      text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
      label_column: `string`, name of the column in the tsv file corresponding
        to the label
      data_url: `string`, url to download the zip file from
      data_dir: `string`, the path to the folder containing the tsv files in the
        downloaded zip
      citation: `string`, citation for the data set
      url: `string`, url for information about the data set
      label_classes: `list[string]`, the list of classes if the label is
        categorical. If not provided, then the label will be of type
        `tf.float32`.
      train_shards: `int`, number of shards for the train data set
      process_label: `Function[string, any]`, function taking in the raw value
        of the label and processing it to the form required by the label feature
      **kwargs: keyword arguments forwarded to super.
    """
        # Version history:
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.1: Initial version.
        super(EntityWithNationalityConfig, self).__init__(
            version=tfds.core.Version(
                "1.0.0",
                # experiments={tfds.core.Experiment.S3: False}
            ),
            # supported_versions=[
            #     tfds.core.Version(
            #         "1.0.0",
            #         "New split API (https://tensorflow.org/datasets/splits)"
            #     ),
            # ],
            **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.train_shards = train_shards
        self.process_label = process_label


@BaseDataset.register("entity_with_nationality")
class EntityWithNationality(BaseDataset):
    """The Chinese General Language Understanding Evaluation (GLUE) benchmark."""

    BUILDER_CONFIGS = [
        EntityWithNationalityConfig(
            name="entity_with_nationality",
            description="""实体名国籍分类""",
            text_features={"entity": "entity"},
            label_classes=["0", "1"],
            label_column="label",
            data_url="",
            data_dir=".",
            citation="TODO",
            url=""
        ),
    ]

    def __init__(self, data_dir, **kwargs):
        super(EntityWithNationality, self).__init__(data_dir, **kwargs)
        if "dataset" in self.hparams and "transformer" in self.hparams.dataset and self.hparams.dataset.transformer is not None:
            self.transformer = Transformer.BaseTransformer.\
                by_name(self.hparams.dataset.transformer)(self.hparams, data_dir=data_dir)
            # data_train_json = transformer.transform(data_train_json, split="train")
            # data_validation_json = transformer.transform(data_validation_json, split="validation")
            # data_test_json = transformer.transform(data_test_json, split="test")

    def _info(self):
        features = self._get_feature_dict()
        if not features:
            logger.warning("Do not specify inputs and outputs in config, using default feature dict.")
            features = self._base_feature_dict()

        metadata = None
        if "dataset" in self.hparams and "tokenizer" in self.hparams.dataset and "name" in self.hparams.dataset.tokenizer:
            metadata = tfds.core.MetadataDict({"tokenizer": self.hparams.dataset.tokenizer.name,
                                               "vocab_size": self.hparams.pretrained.config.vocab_size})

        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict(features),
            metadata=metadata,
            homepage="https://www.cluebenchmarks.com",
            citation=self.builder_config.citation + "\n" + _GLUE_CITATION,
        )

    def _base_feature_dict(self):
        features = {
            text_feature: tfds.features.Text()
            for text_feature in six.iterkeys(self.builder_config.text_features)
        }
        if self.builder_config.label_classes:
            features["label"] = tfds.features.ClassLabel(
                names=self.builder_config.label_classes)
        return features

    def _split_generators(self, dl_manager):
        data_dir = "/search/odin/yyk/workspace/AiSpace/data/downloads/extracted/entities_with_nationality"
        data_train_json, data_validation_json, data_test_json = \
            os.path.join(data_dir, "train.txt"), os.path.join(data_dir, "dev.txt"), os.path.join(data_dir,
                                                                                                   "test.txt")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=self.builder_config.train_shards,
                gen_kwargs={"filepath": data_train_json, 'split': "train"}
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=1,
                gen_kwargs={"filepath": data_validation_json, 'split': "validation"}
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs={"filepath": data_test_json, 'split': "test"}
            )
        ]

    def _generate_examples(self, filepath, **kwargs):
        """
        TODO 直接从原始数据到tfrecords, 不用生成中间的json文件
        :param filepath:
        :param kwargs:
        :return:
        """
        generator = self._generate_examples_from_json if "dataset" in self.hparams and \
                                                         "transformer" in self.hparams.dataset \
                                                         and self.hparams.dataset.transformer is not None \
            else self._generate_examples_from_raw
        for idx, item in enumerate(generator(filepath, **kwargs)):
            yield idx, item

    def _generate_examples_from_raw(self, filepath, **kwargs):
        pass