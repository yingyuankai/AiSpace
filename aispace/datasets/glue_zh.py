# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:24
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : glue_zh.py

"""The Chinese General Language Understanding Evaluation (GLUE) benchmark."""

__all__ = ["GlueZh"]

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
    """BuilderConfig for GLUE."""

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
        super(GlueConfig, self).__init__(
            version=tfds.core.Version(
                "4.0.0",
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


@BaseDataset.register("glue_zh")
class GlueZh(BaseDataset):
    """The Chinese General Language Understanding Evaluation (GLUE) benchmark."""

    BUILDER_CONFIGS = [
        GlueConfig(
            name="afqmc",
            description="""
            Ant Financial Question Matching Corpus, 蚂蚁金融语义相似度。
            每一条数据有三个属性，从前往后分别是 句子1，句子2，句子相似度标签。
            其中label标签，1 表示sentence1和sentence2的含义类似，0表示两个句子的含义不同。""",
            text_features={"sentence1": "sentence1", "sentence2": "sentence2"},
            label_classes=["0", "1"],
            label_column="label",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip",
            data_dir=".",
            citation="TODO",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="tnews",
            description="""
            TNEWS 今日头条中文新闻（短文）分类。
            每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。""",
            text_features={"sentence": "sentence", "keywords": "keywords"},
            label_classes=["news_story", "news_culture", "news_entertainment", "news_sports", "news_finance",
                           "news_house", "news_car", "news_edu", "news_tech", "news_military", "news_travel",
                           "news_world", "news_stock", "news_agriculture", "news_game"],
            label_column="label_desc",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip",
            data_dir=".",
            citation="TODO",
            train_shards=10,
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="iflytek",
            description="""
            IFLYTEK' 长文本分类。该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，
            共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,
            "女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。每一条数据有三个属性，
            从前往后分别是 类别ID，类别名称，文本内容。
            """,
            text_features={"sentence": "sentence"},
            label_classes=[str(i) for i in range(119)],
            label_column="label",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip",
            data_dir=".",
            citation="TODO",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="cmnli",
            description="""
            CMNLI 语言推理任务。CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等，
            对原始MNLI数据和XNLI数据进行了中英文转化，保留原始训练集，合并XNLI中的dev和MNLI中的matched作为CMNLI的dev，
            合并XNLI中的test和MNLI中的mismatched作为CMNLI的test，并打乱顺序。该数据集可用于判断给定的两个句子之间属于蕴涵、
            中立、矛盾关系。每一条数据有三个属性，从前往后分别是 句子1，句子2，蕴含关系标签。其中label标签有三种：neutral，
            entailment，contradiction。
            """,
            text_features={"sentence1": "sentence1", "sentence2": "sentence2"},
            label_classes=["neutral", "entailment", "contradiction"],
            label_column="gold_label",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip",
            data_dir=".",
            citation="TODO",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig( # TODO
            name="copa",
            description="""
            COPA 因果推断-中文版。自然语言推理的数据集，给定一个假设以及一个问题表明是因果还是影响，并从两个选项中选择合适的一个。
            遵照原数据集，我们使用了acc作为评估标准。其中label的标注，0表示choice0，1 表示choice1。原先的COPA数据集是英文的，
            我们使用机器翻译以及人工翻译的方法，并做了些微的用法习惯上的调整，并根据中文的习惯进行了标注，得到了这份数据集。""",
            text_features={"premise": "premise", "choice0": "choice0", "choice1": "choice1", "question": "question"},
            label_classes=[0, 1],
            label_column="label",
            data_url="TODO",
            data_dir=".",
            citation="TODO",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="wsc",
            description="""
            WSC Winograd模式挑战中文版。威诺格拉德模式挑战赛是图灵测试的一个变种，旨在判定AI系统的常识推理能力。
            参与挑战的计算机程序需要回答一种特殊但简易的常识问题：代词消歧问题，即对给定的名词和代词判断是否指代一致。
            其中label标签，true表示指代一致，false表示指代不一致。""",
            text_features={"text": "text", "target": "target"},
            label_classes=["true", "false"],
            label_column="label",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/wsc_public.zip",
            data_dir=".",
            citation="TODO",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="csl",
            description="""
                CSL 论文关键词识别。中文科技文献数据集包含中文核心论文摘要及其关键词。 用tf-idf生成伪造关键词与论文真实关键词混合，
                生成摘要-关键词对，关键词中包含伪造的则标签为0。每一条数据有四个属性，从前往后分别是 数据ID，论文摘要，关键词，真假标签。""",
            text_features={"abst": "abst", "keyword": "keyword"},
            label_classes=["0", "1"],
            label_column="label",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip",
            data_dir=".",
            citation="TODO",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="cmrc2018",
            description="""
                CMRC2018 Reading Comprehension for Simplified Chinese 简体中文阅读理解任务。
                第二届“讯飞杯”中文机器阅读理解评测 (CMRC 2018)""",
            text_features={
                "title": "title",
                "id": "id",
                "context": "context",
                "question": "question",
                "answers": "answers"
            },
            label_column=None,
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip",
            data_dir=".",
            citation="""
            @article{cmrc2018-dataset,
              title={A Span-Extraction Dataset for Chinese Machine Reading Comprehension},
              author={Cui, Yiming and Liu, Ting and Xiao, Li and Chen, Zhipeng and Ma, Wentao and Che, 
              Wanxiang and Wang, Shijin and Hu, Guoping},
              journal={arXiv preprint arXiv:1810.07366},
              year={2018}
            }""",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(
            name="drcd",
            description="""
                DRCD 繁体阅读理解任务, Reading Comprehension for Traditional Chinese。
                台達閱讀理解資料集 Delta Reading Comprehension Dataset (DRCD)(https://github.com/DRCKnowledgeTeam/DRCD) 
                屬於通用領域繁體中文機器閱讀理解資料集。 本資料集期望成為適用於遷移學習之標準中文閱讀理解資料集。 
                数据格式和squad相同，如果使用简体中文模型进行评测的时候可以将其繁转简(本项目已提供)""",
            text_features={
                "title": "title",
                "id": "id",
                "context": "context",
                "question": "question",
                "answers": "answers"
            },
            label_column=None,
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/drcd_public.zip",
            data_dir=".",
            citation="""
                        @article{DBLP:journals/corr/abs-1806-00920,
                          author    = {Chih{-}Chieh Shao and
                                       Trois Liu and
                                       Yuting Lai and
                                       Yiying Tseng and
                                       Sam Tsai},
                          title     = {{DRCD:} a Chinese Machine Reading Comprehension Dataset},
                          journal   = {CoRR},
                          volume    = {abs/1806.00920},
                          year      = {2018},
                          url       = {http://arxiv.org/abs/1806.00920},
                          archivePrefix = {arXiv},
                          eprint    = {1806.00920},
                          timestamp = {Mon, 13 Aug 2018 16:48:22 +0200},
                          biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1806-00920},
                          bibsource = {dblp computer science bibliography, https://dblp.org}
                        }""",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig( # TODO
            name="chid",
            description="""
                ChID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test。
                成语完形填空，文中多处成语被mask，候选项中包含了近义的成语。""",
            data_url="https://storage.googleapis.com/cluebenchmark/tasks/chid_public.zip",
            data_dir=".",
            citation="""
                        @article{DBLP:journals/corr/abs-1906-01265,
                          author    = {Chujie Zheng and
                                       Minlie Huang and
                                       Aixin Sun},
                          title     = {ChID: {A} Large-scale Chinese IDiom Dataset for Cloze Test},
                          journal   = {CoRR},
                          volume    = {abs/1906.01265},
                          year      = {2019},
                          url       = {http://arxiv.org/abs/1906.01265},
                          archivePrefix = {arXiv},
                          eprint    = {1906.01265},
                          timestamp = {Thu, 13 Jun 2019 13:36:00 +0200},
                          biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906-01265},
                          bibsource = {dblp computer science bibliography, https://dblp.org}
                        }""",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
        GlueConfig(  # TODO
            name="c3",
            description="""
                C3 中文多选阅读理解 Multiple-Choice Chinese Machine Reading Comprehension。
                中文多选阅读理解数据集，包含对话和长文等混合类型数据集。""",
            data_dir=".",
            citation="""
                        @article{DBLP:journals/corr/abs-1904-09679,
                          author    = {Kai Sun and
                                       Dian Yu and
                                       Dong Yu and
                                       Claire Cardie},
                          title     = {Probing Prior Knowledge Needed in Challenging Chinese Machine Reading
                                       Comprehension},
                          journal   = {CoRR},
                          volume    = {abs/1904.09679},
                          year      = {2019},
                          url       = {http://arxiv.org/abs/1904.09679},
                          archivePrefix = {arXiv},
                          eprint    = {1904.09679},
                          timestamp = {Fri, 26 Apr 2019 13:18:53 +0200},
                          biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1904-09679},
                          bibsource = {dblp computer science bibliography, https://dblp.org}
                        }""",
            url="https://github.com/CLUEbenchmark/CLUE"
        ),
    ]

    def __init__(self, data_dir, **kwargs):
        super(GlueZh, self).__init__(data_dir, **kwargs)
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
        elif self.builder_config.name not in ["cmrc2018", "drcd"]:
            features["label"] = tf.float32
        if self.builder_config.name not in ["cmrc2018", "drcd"]:
            features["idx"] = tf.int32
        if self.builder_config.name == "wsc":
            features["target"] = tfds.features.FeaturesDict({
                "span1_index": tf.int32,
                "span2_index": tf.int32,
                "span1_text": tfds.features.Text(),
                "span2_text": tfds.features.Text()
            })
        if self.builder_config.name == "csl":
            features["keyword"] = tfds.features.Sequence(tfds.features.Text())
        if self.builder_config.name in ["cmrc2018", "drcd"]:
            features["answers"] = tfds.features.Sequence(
                {
                    "text": tfds.features.Text(),
                    "answer_start": tf.int32
                }
            )
        return features

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.builder_config.data_url)
        data_dir = os.path.join(dl_dir, self.builder_config.data_dir)
        data_train_json, data_validation_json, data_test_json = \
            os.path.join(data_dir, "train.json"), os.path.join(data_dir, "dev.json"), os.path.join(data_dir,
                                                                                                   "test.json")
        # if "dataset" in self.hparams and "transformer" in self.hparams.dataset and self.hparams.dataset.transformer is not None:
        #     transformer = Transformer.BaseTransformer.\
        #         by_name(self.hparams.dataset.transformer)(self.hparams, data_dir=data_dir)
        #     data_train_json = transformer.transform(data_train_json, split="train")
        #     data_validation_json = transformer.transform(data_validation_json, split="validation")
        #     data_test_json = transformer.transform(data_test_json, split="test")

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
        process_label = self.builder_config.process_label
        label_classes = self.builder_config.label_classes
        with open(filepath, 'r', encoding="utf8") as inf:
            if self.builder_config.name in ["cmrc2018", "drcd"]:
                cmrc = json.load(inf)
                for article in cmrc["data"]:
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
                for idx, line in enumerate(inf):
                    if self.builder_config.name not in ["cmrc2018"]:
                        row = json.loads(line)
                    else:
                        row = line
                    example = {
                        feat: row[col]
                        for feat, col in six.iteritems(self.builder_config.text_features)
                    }
                    example["idx"] = idx
                    if self.builder_config.label_column is not None:
                        if self.builder_config.label_column in row:
                            label = row[self.builder_config.label_column]
                            # For some tasks, the label is represented as 0 and 1 in the tsv
                            # files and needs to be cast to integer to work with the feature.
                            if label_classes and label not in label_classes:
                                label = int(label) if label else None
                            example["label"] = process_label(label)
                        else:
                            example["label"] = process_label(-1)

                    # Filter out corrupted rows.
                    for value in six.itervalues(example):
                        if value is None:
                            break
                    else:
                        yield example
