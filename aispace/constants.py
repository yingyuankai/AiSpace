# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-04 15:40
# @Author  : yingyuankai@aliyun.com
# @File    : constants.py

import tensorflow as tf
import tensorflow_datasets as tfds

VERSION = "0.1.0"
LOGO_STR = '\n'.join(
    [
        '',
        r'    ____     _  ______                                     ',
        r'   / /\ \   (_/  ___  |_______    _____     ______   ____  ',
        r'  / /--\ \  | \___   \   ____  \/  ___  \ /  _____| / ___\ ',
        r' / /----\ \ | |____\  \ |____/ |  /___\ |_| |_____| \_____ ',
        r'/_/      \_\|_|______/  |_____/ \_________\ ______\______/ ',
        r'                      | |                                  ',
        r'                      |_|                                  ',
        f'AiSpace v{VERSION}',
        ''
    ]
)

TRAIN_STAGE = "train"
TEST_STAGE = "test"
DEPLOY_STAGE = "deploy"

# Some names
LOGGER_NAME = 'aispace_logger'
TRAIN_DATA_SYMBOL = 'train'
VALIDATION_DATA_SYMBOL = 'validation'
TEST_DATA_SYMBOL = 'test'
MYSELF_LOSS_PREFIX = 'myself'

# Schedules
TRANSFORM_SCHEDULE = 'transform'

# Feature
LIST_OF_INT = 'LIST_OF_INT'
INT = 'INT'
CLASSLABEL = 'CLASSLABEL'
LIST_OF_CLASSLABEL = 'LIST_OF_CLASSLABEL'
STRING = "STRING"

FEATURE_MAPPING = {
    LIST_OF_INT: tfds.features.Sequence(tf.int32),
    INT: tf.int32,
}

# Task
NER = 'NER'
