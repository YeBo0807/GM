#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
Author: YeBo
Data: 2018/4/13 16:22
Python Version: 3.6
File: MFGMnF(1.1).py
Software: PyCharm
'''

import numpy as np
import math
from pandas import Series
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt


def GM_Modeln(history_data, index, nn=0):

    # 计算数据矩阵B和数据向量Y
    # history_data: 历史数据
    # index: 为数据横坐标
    # nn: 开始n值,python需要序列减1
    # x0: 历史数据array形式
    # x1: 一次ago
    # xx1: GM计算累加值
    #
    n = len(history_data)
    x0 = np.array(history_data)
    history_data_ago = [sum(history_data[0:i + 1]) for i in range(n)]
    x1 = np.array(history_data_ago)
    B = np.zeros([n - 1, 2])
    Y = np.zeros([n - 1, 1])
    for i in range(0, n - 1):
        B[i][0] = -0.5 * (x1[i] + x1[i + 1])
        B[i][1] = 1
        Y[i][0] = x0[i + 1]

    # 计算GM(1,1)微分方程的参数a和b
    A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    a = A[0][0]
    b = A[1][0]

    # 建立灰色预测模型
    xx1 = np.zeros(n)
    xx1[0] = x0[0]  # 第一个值相同
    for i in range(2, n + 1):
        xx1[i - 1] = (x1[nn - 1] - b / a) * math.exp(-a * (i - nn)) + b / a  # 因为pytho序列是从0开始的第一个i-1表示K+1

    print('GM(1,1)模型为:\nX(k) = ', x1[nn - 1] - b / a, 'exp(', -a, '(k-n))', b / a)
    print('GM(1,1)模型计算值为:')
    print(Series(xx1, index=index))

    # 还原预测值
    tmp = np.ones(len(xx1))
    for i in range((len(xx1))):
        if i == 0:
            tmp[i] = x0[i]
        else:
            tmp[i] = xx1[i] - xx1[i - 1]
    x_predition = Series(tmp, index=index)
    print('还原值为:\n')
    print(x_predition)

    # 预测值即预测值精度表
    error = np.ones(n)
    relative_error = np.ones(n)
    for i in range(len(x0)):
        error[i] = x0[i] - x_predition[i]
        relative_error[i] = error[i] / x0[i] * 100
    compare = {'GM(1,1)': xx1, '1—AGO': history_data_ago, 'predict_value': x_predition, 'Reality_value': x0,
               'error': error,
               'relative_error': relative_error}
    x_compare = DataFrame(compare, index=index)
    print('预测值即预测值精度表')
    print(x_compare)

    # 模型检验


    # 计算灰色关联度
    # S0 = 0
    # for i in range(1, n - 1):
    #     S0 += x0[i] - x0[0]
    # S0 += (x0[-1] - x0[0]) / 2
    # S0 = np.abs(S0)
    #
    # SS0 = 0
    # for i in range(1, n - 1):
    #     SS0 += x_return[i] - x_return[0]
    # SS0 += (x_return[-1] - x_return[0]) / 2
    # SS0 = np.abs(SS0)
    #
    # S_ = 0
    # for i in range(1, n - 1):
    #     S_ += (x_return[i] - x_return[0]) - (x0[i] - x0[0])
    # S_ += (x_return[-1] - x_return[0] - (x0[-1] - x0[0])) / 2
    # S_ = np.abs(SS0)
    # T = (1 + S0 + SS0) / (1 + S0 + SS0 + S_)
    # if T >= 0.9:
    #     print('精度为一级')
    #     print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)
    # elif T >= 0.8:
    #     print('精度为二级')
    #     print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)
    # elif T >= 0.7:
    #     print('精度为三级')
    #     print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)
    # elif T >= 0.6:
    #     print('精度为四级')
    #     print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)

    B1 = pd.DataFrame([x0, x_predition], index=['x0', 'predition'])
    B1 = np.transpose(B1)
    B1.plot()
    plt.show()

    #  Fourier series
    error_predict_1 = Fourierseries(n, 4, error)
    # print(error_predict_1)

    # exponentialsmoothing
    error_predict_2 = exponentialsmoothing(error, error_predict_1)
    x_final_predition = x_predition
    x_final_predition[1:] = x_predition[1:] + error_predict_1 + error_predict_2
    # x_final_predition = np.zeros(n)
    # x_final_predition[0] =  tmp[0]
    # for i in range(1,n):
    #     x_final_predition[i] = tmp[i] +  error_predict_1[i-1] + error_predict_2[i-1]

    B2 = pd.DataFrame([x0, x_final_predition], index=['x0', 'x_final_predition'])
    B2 = np.transpose(B2)
    # B2.plot()
    # plt.show()

def Fourierseries(n, ka, error):

    # 傅里叶变换模拟一阶误差
    # n: 数据数量
    # ka: a&b总数，2的倍数
    # error: 第一次预测error误差
    error = np.delete(error, 0)        #按照公式这里error去除第一项
    T = n-1
    P = np.zeros([n-1, ka+1])
    P[:, 0] = 1/2  # 赋值第一列
    for i in range(2,n+1):
        kaa = 1
        for j in range(1,ka,2):  # 采用2的间隔，所以1 3 5 7 9
            P[i-2, j] = math.cos(2*math.pi*kaa/T*i)
            P[i-2, j+1] = math.sin(2 * math.pi * kaa/T*i)
            kaa += 1
    C = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(error)

    # 给出error模拟计算公式
    error_predict_1 = np.ones(n-1)
    for k in range(2, n+1):
        error_predict_1[k-2] = 0     #初始化预测值
        ii = 1
        for i in range(1, ka, 2):

            error_predict_1[k-2] = error_predict_1[k-2] + C[i]*math.cos(2*math.pi*ii/T*k) + C[i+1]*math.sin(2*math.pi*ii/T*k)
            ii += 1
        error_predict_1[k-2] += 1/2*C[0]

    return error_predict_1


def exponentialsmoothing(error, error_predict_1, alpha=0.5):

    # 采用一次平滑法
    # error: 一阶实际误差，
    # error_predict_1: 一阶模拟误差
    # alpha: 加权权重值
    error = np.delete(error, 0) # 为了方便比对error,删除error第一项
    n = len(error)
    error_2 = error - error_predict_1
    S = np.zeros(n)
    S[0] = error_2[0]   #初始化状态
    for i in range(1, n):
        S[i] = alpha*error_2[i] + (1-alpha)*S[i-1]
    return S


if __name__ == '__main__':
    # 初始化原始数据
    history_data = [132, 92, 118, 130, 187]
    index = pd.period_range('2001', '2005', freq='A-DEC')
    GM_Modeln(history_data, index, 2)