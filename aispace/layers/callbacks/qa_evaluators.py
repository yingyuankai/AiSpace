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
    def __init__(self, validation_dataset, validation_steps):
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        try:
            validation_results = self.model.predict(self.validation_dataset, steps=self.validation_steps)
        except Exception as e:
            print(e)
        print(len(validation_results))
        print(validation_results)
