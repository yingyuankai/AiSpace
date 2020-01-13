# -*- coding: utf-8 -*-
# @Time    : 2019-11-15 21:09
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : print_lr.py

import tensorflow as tf

# Callback for printing the LR at the end of each epoch.
# class PrintLR(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
#                                                       model.optimizer.lr.numpy()))