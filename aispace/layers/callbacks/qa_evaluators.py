# -*- coding: utf-8 -*-
# @Time    : 2020-07-30 15:06
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : qa_evaluators.py


import tensorflow as tf


class EvaluatorForQaWithImpossible(tf.keras.callbacks.Callback):
    """

    ref: https://keras.io/examples/nlp/text_extraction_with_bert/
    """
    def __init__(self, validation_dataset):
        self.validation_dataset = validation_dataset

    def on_epoch_end(self, epoch, logs=None):

        for itm in self.validation_dataset:
            print(itm)
            break
        # validation_results = self.model.predict(self.validation_dataset)