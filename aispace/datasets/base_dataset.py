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
            if itm.get('type') in [INT, LIST_OF_INT]:
                field_types[itm.get('name')] = FEATURE_MAPPING[itm.get('type')]
            elif itm.get('type') == CLASSLABEL:
                field_types[itm.get('name')] = \
                    tfds.features.ClassLabel(
                        names=itm.get('labels')
                    )
            elif itm.get('type') == LIST_OF_CLASSLABEL:
                field_types[itm.get('name')] = \
                    tfds.features.Sequence(tfds.features.ClassLabel(
                        names=itm.get('labels')
                    ))
        return field_types

    def _generate_examples_from_json(self, filepath, **kwargs):
        import json
        logger.info(f'generating examples from {filepath}')
        fields = [(itm.get('name'), itm.get('column', itm.get('name')))
                  for itm in self.hparams.dataset.inputs + self.hparams.dataset.outputs]
        with open(filepath, 'r') as inf:
            for line in inf:
                row = json.loads(line)
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
