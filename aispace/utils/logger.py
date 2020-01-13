# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-03 14:04
# @Author  : yingyuankai@aliyun.com
# @File    : logger.py

import os


def setup_logging(log_folder, logging_config):
    import logging.config
    # update logging filename
    info_filename = logging_config.handlers.info_file_handler.filename
    error_filename = logging_config.handlers.error_file_handler.filename
    logging_config.handlers.info_file_handler.filename = os.path.join(log_folder, info_filename)
    logging_config.handlers.error_file_handler.filename = os.path.join(log_folder, error_filename)
    # setup logging config
    logging.config.dictConfig(logging_config)