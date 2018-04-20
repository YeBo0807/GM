#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Author: YeBo
Data: 2018/4/17 14:54
Python Version: 3.6
File: MFGMnF_3.py
Software: PyCharm
'''

import numpy as np
import math
from pandas import Series
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 面向对象


class MFGMnF:

    def __init__(self, history_data):
        # 计算数据矩阵B和数据向量Y
        # history_data: 历史数据
        # nn: 开始n值,python需要序列减1,默认为0

        self.history_data = np.array(history_data)
        self.history_data_ago = np.zeros(len(self.history_data))

        self.a = 1
        self.b = 1
        self.x_predition = []
        self.n_prediction = len(self.history_data) + 1

    def GMn_prediction(self):


        xx1 = np.zeros(self.n_prediction)
        xx1[0] = self.history_data[0]  # 第一个值相同
        for i in range(2, self.n_prediction + 1):
            xx1[i - 1] = (self.history_data_ago[self.nn - 1] - self.b / self.a) * math.exp(-self.a * (i - self.nn)) + self.b / self.a
            # 因为pytho序列是从0开始的第一个i-1表
            # 还原预测值
        self.x_predition = np.ones(len(xx1))
        for i in range((len(xx1))):
            if i == 0:
                self.x_predition[i] = self.history_data[i]
            else:
                self.x_predition[i] = xx1[i] - xx1[i - 1]

        # print(self.x_predition)

        return self.x_predition

    def GMn(self,nn, n_prediction):
        # x0: 历史数据array形式
        # x1: 一次ago
        # xx1: GM计算累加值
        self.nn = nn
        self.n_prediction = n_prediction
        n = len(self.history_data)
        self.history_data = np.array(self.history_data)
        self.history_data_ago = np.array([sum(self.history_data[0:i + 1]) for i in range(n)])
        B = np.zeros([n - 1, 2])
        Y = np.zeros([n - 1, 1])
        for i in range(0, n - 1):
            B[i][0] = -0.5 * (self.history_data_ago[i] + self.history_data_ago[i + 1])
            B[i][1] = 1
            Y[i][0] = self.history_data[i + 1]
        # 计算GM(1,1)微分方程的参数a和b
        A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
        self.a = A[0][0]
        self.b = A[1][0]

        # 进行GMn预测
        self.x_predition = self.GMn_prediction()
        error = np.ones(n)
        for i in range(n):
            error[i] = self.history_data[i] - self.x_predition[i]

        error_predict_1 = self.Fourierseries(error)
        error_predict_2 = self.exponentialsmoothing(error, error_predict_1)
        # print(self.x_predition[1:self.n_prediction])
        # print(error_predict_1[0:self.n_prediction-1])
        # print(error_predict_2[0:self.n_prediction-1])
        x_prediction_final = np.zeros(len(self.x_predition)) # 初始化 final
        x_prediction_final[0] = self.history_data[0]
        error_predict_1 = np.insert(error_predict_1,0,0)
        error_predict_2 = np.insert(error_predict_2, 0, 0)
        x_prediction_final = self.x_predition + error_predict_1 + error_predict_2

        return x_prediction_final, self.x_predition, error_predict_1, error_predict_2


    def Fourierseries(self, error, ka=4):
        # 傅里叶变换模拟一阶误差
        # n: 数据数量
        # ka: a&b总数，2的倍数
        # error: 第一次预测error误差
        n = len(self.history_data)
        error = np.delete(error, 0)
        T = n - 1
        P = np.zeros([n - 1, ka + 1])
        P[:, 0] = 1 / 2  # 赋值第一列
        for i in range(2, n + 1):
            kaa = 1
            for j in range(1, ka, 2):  # 采用2的间隔，所以1 3 5 7 9
                P[i - 2, j] = math.cos(2 * math.pi * kaa / T * i)
                P[i - 2, j + 1] = math.sin(2 * math.pi * kaa / T * i)
                kaa += 1
        C = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(error)
        error_predict_1 = np.ones(self.n_prediction - 1)
        for k in range(2, self.n_prediction + 1):
            error_predict_1[k - 2] = 0  # 初始化预测值
            ii = 1
            for i in range(1, ka, 2):
                error_predict_1[k - 2] = error_predict_1[k - 2] + C[i] * math.cos(2 * math.pi * ii / T * k) + \
                                         C[i + 1] * math.sin(2 * math.pi * ii / T * k)
                ii += 1
            error_predict_1[k - 2] += 1 / 2 * C[0]

        return error_predict_1

    def exponentialsmoothing(self, error, error_predict_1, alpha=0.5):


        # 采用一次平滑法
        # error: 一阶实际误差，
        # error_predict_1: 一阶模拟误差
        # alpha: 加权权重值
        error = np.delete(error, 0)  # 为了方便比对error,删除error第一项
        n = len(error)
        error_predict_1 = np.delete(error_predict_1, list(range(n,self.n_prediction-1))) #因为预测error多了一项
        error_2 = error - error_predict_1
        error_predict_2 = np.zeros(self.n_prediction-1)
        error_predict_2[0] = error_2[0]
        for i in range(0, self.n_prediction-1):
            if i == 0:
                error_predict_2[i] = alpha * error_2[0] + (1 - alpha) * error_predict_2[0]
            else:
                error_predict_2[i] = alpha * error_2[i-1] + (1 - alpha) * error_predict_2[i-1]  # 这部导致了其只能预测下一个输入，不能多个

        return error_predict_2



if __name__ == '__main__':
    history_data = [132, 92, 118, 130, 187]
    nn = MFGMnF(history_data)
    x_prediction_final, x_predition, error_predict_1, error_predict_2 = nn.GMn(3, 6)
    print(x_prediction_final, '\n' ,x_predition, '\n' ,error_predict_1, '\n' ,error_predict_2)


    threshold = 1.0e-2
    # x1_data = np.random.randn(100).astype(np.float32)
    # x2_data = np.random.randn(100).astype(np.float32)
    # x3_data = np.random.randn(100).astype(np.float32)
    # y_data = x1_data*2 + x2_data*3 + x3_data*4 + 1.5

    weight1 = tf.Variable(1.)
    weight2 = tf.Variable(1.)
    weight3 = tf.Variable(1.)
    bias = tf.constant([0.])

    x1_ = tf.placeholder(tf.float32)
    x2_ = tf.placeholder(tf.float32)
    x3_ = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)

    y_model = tf.add(tf.add(tf.multiply(x1_, weight1), tf.add(tf.multiply(x2_, weight2), tf.multiply(x3_, weight3))), bias)
    loss = tf.reduce_mean(tf.pow((y_model - y_), 2))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    flag = 1
    while(flag):
        for (x,y) in zip(zip(x_predition, error_predict_1, error_predict_2), x_prediction_final):
            sess.run(train_op, feed_dict={x1_: x[0], x2_: x[1], x3_: x[2], y_: y})
        print('weihgt1: ', weight1.eval(sess), 'weihgt2: ', weight2.eval(sess), 'weihgt3: ', weight3.eval(sess))

        if sess.run(loss, feed_dict={x1_: x[0], x2_: x[1], x3_: x[2], y_: y}) <= threshold:
            flag = 0

    print('weihgt1: ', weight1.eval(sess), 'weihgt2: ', weight2.eval(sess), 'weihgt3: ', weight3.eval(sess))


    #
    # threshold = 1.0e-2
    # x1_data = np.random.randn(100).astype(np.float32)
    # x2_data = np.random.randn(100).astype(np.float32)
    #
    # y_data = x1_data*2 + x2_data*3  + 1.5
    #
    # weight1 = tf.Variable(1.)
    # weight2 = tf.Variable(1.)
    #
    # bias = tf.Variable(1.)
    #
    # x1_ = tf.placeholder(tf.float32)
    # x2_ = tf.placeholder(tf.float32)
    #
    # y_ = tf.placeholder(tf.float32)
    #
    # y_model = tf.add(tf.add(tf.multiply(x1_, weight1), tf.multiply(x2_, weight2)), bias)
    # loss = tf.reduce_mean(tf.pow((y_model - y_), 2))
    #
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # flag = 1
    # while(flag):
    #     for (x,y) in zip(zip(x1_data, x2_data), y_data):
    #         sess.run(train_op, feed_dict={x1_: x[0], x2_: x[1], y_: y})
    #     if sess.run(loss, feed_dict={x1_: x[0], x2_: x[1], y_: y}) <= threshold:
    #         flag = 0
    #
    # print('weihgt1: ', weight1.eval(sess), 'weihgt2: ', weight2.eval(sess),'bias: ', bias.eval(sess))




