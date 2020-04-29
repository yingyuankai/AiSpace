# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-09 15:18
# @Author  : yingyuankai@aliyun.com
# @File    : base_dataset.py

import logging
from abc import ABCMeta, abstractmethod

from aispace.utils.registry import Registry
from aispace.constants import *

from aispace.utils.hparams import Hparams


__all__ = [
    'BaseDataset'
]

logger = logging.getLogger(__name__)


class BaseDataset(Registry, tfds.core.GeneratorBasedBuilder):
    __metaclass__ = ABCMeta

    def __init__(self, data_dir, **kwargs):
        self.hparams = kwargs.pop("hparams")
        # data_dir = self.hparams.dataset.data_dir
        super(BaseDataset, self).__init__(data_dir=data_dir, **kwargs)

    def _get_feature_dict(self):
        field_types = {}
        feature_labels = []
        if 'dataset' in self.hparams:
            if 'inputs' in self.hparams.dataset:
                feature_labels.extend(self.hparams.dataset.inputs)
            if 'outputs' in self.hparams.dataset:
                feature_labels.extend(self.hparams.dataset.outputs)
        for itm in feature_labels:
            # Read labels from specify file when have labels too much
            cur_labels = itm.get("labels")
            if "labels" in itm and \
                    not isinstance(itm.get("labels"), (list, tuple)) and \
                    "url" in itm.get("labels"):
                logger.info(f'Load labels from file: ${itm.get("labels", {}).get("url")}')
                cur_label_map = getattr(self.transformer,
                                        itm.get("labels", {}).get("name", "prepare_labels"),
                                        self.transformer.prepare_labels)(itm.get("labels", {}).get("url"),
                                                                         itm.get("labels", {}).get("name", ""))
                # For convenience, keep this vocab on the first level
                self.hparams.cascade_set(itm.get("labels", {}).get("name", "label_vocab"), cur_label_map)
                cur_labels = list(cur_label_map.values())
                itm.cascade_set("labels", cur_labels)
                itm.cascade_set("num", len(cur_labels))
                # Replace feature values of same name with itm
                if 'dataset' in self.hparams:
                    if 'inputs' in self.hparams.dataset:
                        new_tmp = []
                        flag = False
                        for tmp in self.hparams.dataset.inputs:
                            if itm.name == tmp.name:
                                new_tmp.append(itm)
                                flag = True
                            else:
                                new_tmp.append(tmp)
                        if flag:
                            self.hparams.cascade_set("dataset.inputs", new_tmp)
                    if 'outputs' in self.hparams.dataset:
                        new_tmp = []
                        flag = False
                        for tmp in self.hparams.dataset.outputs:
                            if itm.name == tmp.name:
                                new_tmp.append(itm)
                                flag = True
                            else:
                                new_tmp.append(tmp)
                        if flag:
                            self.hparams.cascade_set("dataset.outputs", new_tmp)

            if itm.get('type') in [INT, LIST_OF_INT]:
                field_types[itm.get('name')] = FEATURE_MAPPING[itm.get('type')]
            elif itm.get('type') == CLASSLABEL:
                field_types[itm.get('name')] = \
                    tfds.features.ClassLabel(
                        names=cur_labels
                    )
            elif itm.get('type') == LIST_OF_CLASSLABEL:
                field_types[itm.get('name')] = \
                    tfds.features.Sequence(tfds.features.ClassLabel(
                        names=cur_labels
                    ))
        return field_types

    def _generate_examples_from_json(self, filepath, **kwargs):
        import json
        logger.info(f'generating examples from {filepath}')
        fields = [(itm.get('name'), itm.get('column', itm.get('name')))
                  for itm in self.hparams.dataset.inputs + self.hparams.dataset.outputs]
        with open(filepath, 'r') as inf:
            for line in inf:
                try:
                    row = json.loads(line)
                except:
                    logger.error("Read json Err!", exc_info = True)
                instance_final = {}
                for field_name, field_column in fields:
                    field_value = row.get(field_column)
                    if field_value is None:
                        continue
                    if isinstance(field_value, str):
                        try:
                            field_value = eval(field_value)
                        except NameError:
                            pass
                        finally:
                            field_value = field_value

                    instance_final[field_name] = field_value
                yield instance_final
