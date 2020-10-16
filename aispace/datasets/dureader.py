# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:24
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : glue_zh.py

"""DuReader."""

__all__ = [
    "Dureader"
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

ROBUST_NAME = 'robust'
YESNO_NAME = 'yesno'


class DuReaderConfig(tfds.core.BuilderConfig):
    """BuilderConfig for DuReader."""

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
        super(DuReaderConfig, self).__init__(
            version=tfds.core.Version(
                "3.0.0",
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


@BaseDataset.register("dureader")
class Dureader(BaseDataset):
    """DuReader"""

    BUILDER_CONFIGS = [

        DuReaderConfig(
            name=ROBUST_NAME,
            description="""
                阅读理解模型的鲁棒性是衡量该技术能否在实际应用中大规模落地的重要指标之一。
                随着当前技术的进步，模型虽然能够在一些阅读理解测试集上取得较好的性能，但在实际应用中，这些模型所表现出的鲁棒性仍然难以令人满意。
                DuReaderrobust数据集作为首个关注阅读理解模型鲁棒性的中文数据集，旨在考察模型在真实应用场景中的过敏感性、过稳定性以及泛化能力等问题。""",
            text_features={
                "title": "title",
                "id": "id",
                "context": "context",
                "question": "question",
                "answers": "answers"
            },
            label_column=None,
            data_url="https://dataset-bj.cdn.bcebos.com/qianyan/dureader_robust-data.tar.gz",
            data_dir=".",
            citation="""
            @article{tang2020dureaderrobust,
                title={DuReaderrobust: A Chinese Dataset Towards Evaluating the Robustness of Machine Reading Comprehension Models},
                author={Tang, Hongxuan and Liu, Jing and Li, Hongyu and Hong, Yu and Wu, Hua and Wang, Haifeng},
                journal={arXiv preprint arXiv:2004.11142},
                year={2020}
            }""",
            url="https://aistudio.baidu.com/aistudio/competition/detail/49"
        ),
        DuReaderConfig(
            name=YESNO_NAME,
            description="""
            机器阅读理解评测中常用的F1、EM等指标虽然能够很好的衡量抽取式模型所预测的答案和真实答案的匹配程度，但在处理观点类问题时，该类指标难以衡量模型是否真正理解答案所代表的含义，例如答案中包含的观点极性。DuReaderyesno是一个以观点极性判断为目标任务的数据集，通过引入该数据集，可以弥补抽取类数据集的不足，从而更好地评价模型的自然语言理解能力。

            该数据集的任务定义如下：
            对于一个给定的问题q、一系列相关文档D=d1, d2, …, dn，以及人工抽取答案段落摘要a，要求参评系统自动对问题q、候选文档D以及答案段落摘要a进行分析，输出每个答案段落摘要所表述的是非观点极性。其中，极性分为三类 {Yes, No, Depends}。其中：

            Yes：肯定观点，肯定观点指的是答案给出了较为明确的肯定态度。有客观事实的从客观事实的角度出发，主观态度类的从答案的整体态度来判断。
            No：否定观点，否定观点通常指的是答案较为明确的给出了与问题相反的态度。
            Depends：无法确定/分情况，主要指的是事情本身存在多种情况，不同情况下对应的观点不一致；或者答案本身对问题表示不确定，要具体具体情况才能判断。""",
            text_features={
                "id": "id",
                "documents": "documents",
                "question": "question",
                "answer": "answer",
                "yesno_answer": "yesno_answer"
            },
            label_column=None,
            data_url="https://dataset-bj.cdn.bcebos.com/qianyan/dureader_yesno-data.tar.gz",
            data_dir=".",
            citation="""
            @article{tang2020dureaderrobust,
            title={DuReaderrobust: A Chinese Dataset Towards Evaluating the Robustness of Machine Reading Comprehension Models},
            author={Tang, Hongxuan and Liu, Jing and Li, Hongyu and Hong, Yu and Wu, Hua and Wang, Haifeng},
            journal={arXiv preprint arXiv:2004.11142},
            year={2020}
            }""",
            url="https://aistudio.baidu.com/aistudio/competition/detail/49"
        ),
    ]

    def __init__(self, data_dir, **kwargs):
        super(Dureader, self).__init__(data_dir, **kwargs)
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
            homepage="https://aistudio.baidu.com/aistudio/competition/detail/49",
            citation=self.builder_config.citation + "\n" + _GLUE_CITATION,
        )

    def _base_feature_dict(self):
        features = {
            text_feature: tfds.features.Text()
            for text_feature in six.iterkeys(self.builder_config.text_features)
        }

        if self.builder_config.name == ROBUST_NAME:
            features["answers"] = tfds.features.Sequence(
                {
                    "text": tfds.features.Text(),
                    "answer_start": tf.int32
                }
            )
        else:
            features['documents'] = tfds.features.Sequence(
                {
                    "title": tfds.features.Text(),
                    "paragraphs": tfds.features.Sequence(tfds.features.Text())
                }
            )
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
        with open(filepath, 'r', encoding="utf8") as inf:
            if self.builder_config.name == ROBUST_NAME:
                one_doc = json.load(inf)
                for article in one_doc["data"]:
                    title = article.get("title", "").strip()
                    for paragraph in article["paragraphs"]:
                        context = paragraph["context"].strip()
                        for qa in paragraph["qas"]:
                            question = qa["question"].strip()
                            id_ = qa["id"]

                            answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                            answers = [answer["text"].strip() for answer in qa["answers"]]

                            # Features currently used are "context", "question", and "answers".
                            # Others are extracted here for the ease of future expansions.
                            yield {
                                "title": title,
                                "context": context,
                                "question": question,
                                "id": id_,
                                "answers": {
                                    "answer_start": answer_starts,
                                    "text": answers,
                                },
                            }
            else:
                for line in inf:
                    one_doc = json.loads(line)
                    one_doc['id'] = str(one_doc['id'])
                    yield one_doc
