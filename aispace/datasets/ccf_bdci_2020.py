# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:24
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : glue_zh.py

"""DuReader."""

__all__ = [
    "Ccfbdci2020"
]

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


class CcfBdci2020Config(tfds.core.BuilderConfig):
    """BuilderConfig for DuReader."""

    # @tfds.core.disallow_positional_args
    def __init__(self,
                 text_features=None,
                 label_column=None,
                 data_urls=None,
                 data_dir=None,
                 citation=None,
                 url=None,
                 label_classes=None,
                 train_shards=1,
                 process_label=lambda x: x,
                 **kwargs):
        """BuilderConfig for DuReader.

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
        super(CcfBdci2020Config, self).__init__(
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
        self.data_urls = data_urls
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.train_shards = train_shards
        self.process_label = process_label


@BaseDataset.register("ccfbdci2020")
class Ccfbdci2020(BaseDataset):
    """Ccfbdci2020"""

    BUILDER_CONFIGS = [

        CcfBdci2020Config(
            name='plain_text',
            description="""
                闲聊对话相关数据：华为的微博数据 [1] ，北航和微软的豆瓣多轮对话 [2] ，清华的LCCC数据集 [3] 。
                知识对话相关数据：百度的DuConv [4] ，清华的KdConv [5]，腾讯的检索辅助生成对话数据集 [6]。
                推荐对话相关数据：百度的DuRecDial [7]。""",
            data_url=["https://dataset-bj.cdn.bcebos.com/qianyan/douban.zip",
                      "https://dataset-bj.cdn.bcebos.com/qianyan/duconv.zip",
                      "https://dataset-bj.cdn.bcebos.com/qianyan/DuRecDial.zip",
                      "https://dataset-bj.cdn.bcebos.com/qianyan/LCCC.zip",
                      "https://dataset-bj.cdn.bcebos.com/qianyan/kdconv.zip",
                      "https://dataset-bj.cdn.bcebos.com/qianyan/tencent.zip",
                      "https://dataset-bj.cdn.bcebos.com/qianyan/weibo.zip"],
            data_dir=".",
            citation="",
            url="https://aistudio.baidu.com/aistudio/competition/detail/49"
        ),
    ]

    def __init__(self, data_dir, **kwargs):
        super(Ccfbdci2020, self).__init__(data_dir, **kwargs)
        if "dataset" in self.hparams and "transformer" in self.hparams.dataset and self.hparams.dataset.transformer is not None:
            self.transformer = Transformer.BaseTransformer.\
                by_name(self.hparams.dataset.transformer)(self.hparams, data_dir=data_dir)

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
            homepage="https://aistudio.baidu.com/aistudio/competition/detail/55",
            citation=self.builder_config.citation + "\n" + _GLUE_CITATION,
        )

    def _base_feature_dict(self):
        features = {
            text_feature: tfds.features.Text()
            for text_feature in six.iterkeys(self.builder_config.text_features)
        }
        return features

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.builder_config.data_url)
        data_dir = os.path.join(dl_dir, self.builder_config.data_dir)

        data_train_json, data_validation_json, data_test_json = \
            os.path.join(data_dir, f"dureader_{self.builder_config.name}-data/train.json"), \
            os.path.join(data_dir, f"dureader_{self.builder_config.name}-data/dev.json"), \
            os.path.join(data_dir, f"dureader_{self.builder_config.name}-data/test.json")
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
        直接从原始数据到tfrecords, 不用生成中间的json文件
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
