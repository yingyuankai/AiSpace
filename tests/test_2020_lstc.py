# -*- coding: utf-8 -*-
# @Time    : 2019-12-23 15:01
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : test_glue.py

import os, sys
import tensorflow as tf
import tensorflow_datasets as tfds
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow_datasets.core.download import DownloadConfig

from aispace.datasets import *
from aispace.utils.hparams import Hparams
from aispace.utils.builder_utils import load_dataset


class TestLSTC(unittest.TestCase):
    def test_lstc_load(self):
        hparams = Hparams()
        hparams.load_from_config_file("../configs/2020_LSTC/DuEE_trigger.yml")
        hparams.stand_by()
        checksum_dir = "../aispace/datasets/url_checksums"
        tfds.download.add_checksums_dir(checksum_dir)
        # download_config = DownloadConfig(register_checksums=True)
        tnews = tfds.load("lstc_2020/DuEE_role",
                          # data_dir="/search/data1/yyk/data/datasets/glue_zh",
                          data_dir="../data",
                          builder_kwargs={'hparams': hparams},
                          # download_and_prepare_kwargs={'download_config': download_config}
                          )

        tokenizer = BertTokenizer(hparams.dataset.tokenizer)
        # id_to_label = {v: k for k, v in hparams.duee_event_type_labels.items()}
        label_counter = {}
        for itm in tnews["validation"]:
            for k, v in itm.items():
                if v.shape[0] == 151:
                    print(itm)
                    break
            # print(itm)
            # print()
            # print(tokenizer.decode(itm["input_ids"].numpy().tolist()))
            # l = hparams.dataset.outputs[0].labels[tf.argmax(itm["output_1"], -1).numpy().tolist()]
            # print(id_to_label[l])
            # if id_to_label[l] not in label_counter:
            #     label_counter[id_to_label[l]] = 0
            # label_counter[id_to_label[l]] += 1
        # print(label_counter)
        # print(len(label_counter))

# python -u aispace/trainer.py \
#    --experiment_name test \
#    --model_name bert_for_classification \
#    --schedule train_and_eval \
#    --config_name tnews \
#    --config_dir ./configs/glue_zh \
#    --gpus 0 1 2 3