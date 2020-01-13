# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 15:15
# @Author  : yingyuankai
# @Email   : yingyuankai@aliyun.com
# @File    : focal_losses.py


import tensorflow as tf

from aispace.utils.tf_utils import get_shape

__all__ = [
    'focal_loss_softmax',
    'focal_loss_sigmoid_v1',
    'focal_loss_sigmoid_v2'
]


def focal_loss_softmax(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed


def focal_loss_sigmoid_v1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    alpha = tf.math.softmax(alpha)
    alpha = tf.expand_dims(alpha, -1)
    gamma = float(gamma)

    def focal_loss_fixed(y_true, y_pred):
        true_rank, pred_rank = y_true.shape.ndims, y_pred.shape.ndims
        true_shape, pred_shape = get_shape(y_true), get_shape(y_pred)
        label_num = get_shape(y_pred)[-1]
        if true_rank + 1 == pred_rank:
            y_true = tf.one_hot(y_true, label_num)
            true_rank = y_true.shape.ndims
        if not tf.is_tensor(true_shape[-1]) and true_shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(tf.reshape(y_true, pred_shape[:-1]), tf.int32), label_num)
        true_shape, pred_shape = get_shape(y_true), get_shape(y_pred)
        assert true_rank == pred_rank, \
            f"For the tensor y_true, the actual tensor rank {true_rank} (shape = {get_shape(y_true)}) " \
                f"is not equal to the expected tensor rank {pred_rank}"
        # assert true_shape == pred_shape, \
        #     f"For the tensor y_true, the actual tensor shape = {true_shape}) " \
        #         f"is not equal to the expected tensor shape {pred_shape}"

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.softmax(y_pred, -1)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.math.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return focal_loss_fixed


def focal_loss_sigmoid_v2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        true_rank, pred_rank = y_true.shape.ndims, y_pred.shape.ndims
        if true_rank + 1 == pred_rank:
            label_num = get_shape(y_pred)[-1]
            y_true = tf.one_hot(y_true, label_num)
            true_rank = y_true.shape.ndims
        assert true_rank == pred_rank, \
            f"For the tensor y_true, the actual tensor rank {true_rank} (shape = {get_shape(y_true)}) " \
                f"is not equal to the expected tensor rank {pred_rank}"
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.math.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return focal_loss_fixed