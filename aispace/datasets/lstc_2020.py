# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:24
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : glue_zh.py

"""The dataset of 2020 Language and Smart Technology Competition"""

__all__ = ["LSTC_2020"]

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


class GlueConfig(tfds.core.BuilderConfig):
    """BuilderConfig for 2020 LSTC."""

    @tfds.core.disallow_positional_args
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
        """BuilderConfig for 2020 LSTC.

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
        super(GlueConfig, self).__init__(
            version=tfds.core.Version(
                "5.0.0",
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


@BaseDataset.register("LSTC_2020")
class LSTC_2020(BaseDataset):
    """The Chinese General Language Understanding Evaluation (GLUE) benchmark."""

    BUILDER_CONFIGS = [
        GlueConfig(
            name="DuEE_trigger",
            description="""
            事件抽取 (Event Extraction, EE)是指从自然语言文本中抽取事件并识别事件类型和事件元素的技术。
            事件抽取是智能风控、智能投研、舆情监控等人工智能应用的重要技术基础，受到学术界和工业界的广泛关注。
            事件抽取任务涉及事件句抽取、触发词识别、事件类型判别、论元抽取等复杂技术，具有一定的挑战。
            Event Extraction is an advanced text analysis technology that identifies events and extracts corresponding 
            event arguments with argument roles from texts. Event extraction is the key technology of many AI 
            applications, such as risk management, investment research and public opinion monitoring, and have received 
            a lot of attention from academia and industry. Event extraction involves several complicated tasks such as 
            event identification, event classification, argument identification and argument classification, 
            which makes it a challenging task.
            本次竞赛将提供业界规模最大的中文事件抽取数据集 DuEE，旨在为研究者提供学术交流平台，进一步提升事件关系抽取技术的研究水平，
            推动相关人工智能应用的发展。
            At this competition, we will release the largest general-purpose Chinese event extraction data set (DuEE), 
            aiming to provide researchers with a platform to communicate and further improve the performance of Chinese 
            event extraction technology and promote the application of event extraction in different areas.""",
            text_features={"text": "text", "id": "id"},
            label_column=None,
            data_url="https://dataset-bj.cdn.bcebos.com/event_extraction",
            data_dir=".",
            citation="TODO",
            url="https://aistudio.baidu.com/aistudio/competition/detail/32"
        ),
        GlueConfig(
            name="DuEE_role",
            description="""
                事件抽取 (Event Extraction, EE)是指从自然语言文本中抽取事件并识别事件类型和事件元素的技术。
                事件抽取是智能风控、智能投研、舆情监控等人工智能应用的重要技术基础，受到学术界和工业界的广泛关注。
                事件抽取任务涉及事件句抽取、触发词识别、事件类型判别、论元抽取等复杂技术，具有一定的挑战。
                Event Extraction is an advanced text analysis technology that identifies events and extracts corresponding 
                event arguments with argument roles from texts. Event extraction is the key technology of many AI 
                applications, such as risk management, investment research and public opinion monitoring, and have received 
                a lot of attention from academia and industry. Event extraction involves several complicated tasks such as 
                event identification, event classification, argument identification and argument classification, 
                which makes it a challenging task.
                本次竞赛将提供业界规模最大的中文事件抽取数据集 DuEE，旨在为研究者提供学术交流平台，进一步提升事件关系抽取技术的研究水平，
                推动相关人工智能应用的发展。
                At this competition, we will release the largest general-purpose Chinese event extraction data set (DuEE), 
                aiming to provide researchers with a platform to communicate and further improve the performance of Chinese 
                event extraction technology and promote the application of event extraction in different areas.""",
            text_features={"text": "text", "id": "id"},
            label_column=None,
            data_url="https://dataset-bj.cdn.bcebos.com/event_extraction",
            data_dir=".",
            citation="TODO",
            url="https://aistudio.baidu.com/aistudio/competition/detail/32"
        )
    ]

    def __init__(self, data_dir, **kwargs):
        # data transformer
        if "dataset" in kwargs["hparams"] and \
                "transformer" in kwargs["hparams"].dataset and \
                kwargs["hparams"].dataset.transformer is not None:
            self.transformer = Transformer.BaseTransformer. \
                by_name(kwargs["hparams"].dataset.transformer)(kwargs["hparams"])

        super(LSTC_2020, self).__init__(data_dir, **kwargs)

    def _info(self):
        features = self._get_feature_dict()
        if not features:
            logger.warning("Do not specify inputs and outputs in config, using default feature dict.")
            features = self._base_feature_dict()

        metadata = None
        if "dataset" in self.hparams and \
                "tokenizer" in self.hparams.dataset and \
                "name" in self.hparams.dataset.tokenizer:
            metadata = tfds.core.MetadataDict({"tokenizer": self.hparams.dataset.tokenizer.name})

        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict(features),
            metadata=metadata,
            homepage="https://aistudio.baidu.com/aistudio/competition",
            citation=self.builder_config.citation + "\n" + _GLUE_CITATION,
        )

    def _base_feature_dict(self):
        features = {
            text_feature: tfds.features.Text()
            for text_feature in six.iterkeys(self.builder_config.text_features)
        }
        if self.builder_config.name in ["DuEE_trigger"]:
            features["triggers"] = tfds.features.Sequence(
                {
                    "event_type": tfds.features.Text(),
                    "trigger": tfds.features.Text(),
                    "trigger_start_index": tf.int32
                }
            )
        elif self.builder_config.name in ["DuEE_role"]:
            features["event_type"] = tfds.features.Text()
            features["trigger"] = tfds.features.Text()
            features["trigger_start_index"] = tf.int32
            features["arguments"] = tfds.features.Sequence(
                {
                    "role": tfds.features.Text(),
                    "argument": tfds.features.Text(),
                    "argument_start_index": tf.int32
                }
            )
        return features

    def _split_generators(self, dl_manager):
        train_dl_dir = dl_manager.download_and_extract(self.builder_config.data_url + "/train_data.json.zip")
        dev_dl_dir = dl_manager.download_and_extract(self.builder_config.data_url + "/dev_data.json.zip")
        test_dl_dir = dl_manager.download_and_extract(self.builder_config.data_url + "/test1_data.json.zip")

        train_dl_dir = os.path.join(train_dl_dir, self.builder_config.data_dir)
        dev_dl_dir = os.path.join(dev_dl_dir, self.builder_config.data_dir)
        # test_dl_dir = os.path.join(test_dl_dir, self.builder_config.data_dir)

        data_train_json, \
        data_validation_json = \
            os.path.join(train_dl_dir, "train_data/train.json"), \
            os.path.join(dev_dl_dir, "dev_data/dev.json"), \

        if "dataset" in self.hparams and \
                "transformer" in self.hparams.dataset and \
                self.hparams.dataset.transformer is not None:
            data_train_json = self.transformer.transform(data_train_json, split="train")
            data_validation_json = self.transformer.transform(data_validation_json, split="validation")
            # data_test_json = self.transformer.transform(data_test_json, split="test")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=self.builder_config.train_shards,
                gen_kwargs={"filepath": data_train_json}
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=1,
                gen_kwargs={"filepath": data_validation_json}
            ),
            # tfds.core.SplitGenerator(
            #     name=tfds.Split.TEST,
            #     num_shards=1,
            #     gen_kwargs={"filepath": data_test_json}
            # )
        ]

    def _generate_examples(self, filepath, **kwargs):
        generator = self._generate_examples_from_json if "dataset" in self.hparams and \
                                                         "transformer" in self.hparams.dataset \
                                                         and self.hparams.dataset.transformer is not None \
            else self._generate_examples_from_raw
        for idx, item in enumerate(generator(filepath, **kwargs)):
            yield idx, item

    def _generate_examples_from_raw(self, filepath, **kwargs):
        with open(filepath, 'r', encoding="utf8") as inf:
            for line in inf:
                itm = json.loads(line)
                if self.builder_config.name.startswith("DuEE"):
                    text = itm.get("text", "")
                    id = itm.get("id", "")
                    event_list = itm.get("event_list", [])
                    trigger_examples = []
                    role_examples = []
                    for event in event_list:
                        event_type = event.get("event_type")
                        trigger = event.get("trigger")
                        trigger_start_index = event.get("trigger_start_index")
                        if event_type is None or trigger is None or trigger_start_index is None:
                            continue
                        tmp = {
                            "event_type": event_type, "trigger": trigger, "trigger_start_index": trigger_start_index
                        }
                        trigger_examples.append(tmp)
                        arguments = []
                        for arg in event.get("arguments", []):
                            del arg["alias"]
                            arguments.append(arg)
                        if len(arguments) == 0:
                            continue
                        if self.builder_config.name in ["DuEE_role"]:
                            one_example = {
                                "text": text,
                                "id": id,
                                "event_type": event_type,
                                "trigger": trigger,
                                "trigger_start_index": trigger_start_index,
                                "arguments": arguments
                            }
                            role_examples.append(one_example)

                    if self.builder_config.name in ["DuEE_trigger"]:
                        one_example = {
                            "text": text, "id": id, "triggers": trigger_examples
                        }
                        yield one_example
                    else:
                        if len(role_examples) == 0:
                            continue
                        for one_example in role_examples:
                            yield one_example
