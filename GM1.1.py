import numpy as np
import math
from pandas import Series
from pandas import DataFrame
import pandas as pd





def GM_Model(history_data, index):
    # 计算数据矩阵B和数据向量Y
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
    xx0 = np.zeros(n)
    xx0[0] = x0[0] # 第一个值相同
    for i in range(1,n+1):
        xx0[i-1] = (x0[0] - b / a) * math.exp(-a * (i - 1)) + b / a  #因为pytho序列是从0开始的

    print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b/a)
    print('GM(1,1)模型计算值为:')
    print(Series(xx0, index=index))

    # 还原预测值
    tmp = np.ones(len(xx0))
    for i in range((len(xx0))):
        if i == 0:
            tmp[i] = xx0[i]
        else:
            tmp[i] = xx0[i] - xx0[i-1]
    X_Return = Series(tmp, index=index)
    print('还原值为:\n')
    print(X_Return)

    # 预测值即预测值精度表
    error = np.ones(n)
    relative_error = np.ones(n)
    for i in range(len(x0)):
        error[i] = x0[i]-X_Return[i]
        relative_error[i] = error[i]/x0[i]*100
    compare = {'GM(1,1)计算值': xx0, '1—AGO': history_data_ago, '预测值': X_Return, '实际值': x0, '误差': error,
               '相对误差(%)': relative_error}
    x_compare = DataFrame(compare, index=index)
    print('预测值即预测值精度表')
    print(x_compare)


    # 模型检验

    error_avg = np.mean(error)    # 平均相对误差

    # 计算灰色关联度
    S0 = 0
    for i in range(1, n-1):
        S0 += x0[i]-x0[0]
    S0 += (x0[-1] - x0[0]) / 2
    S0 = np.abs(S0)

    SS0 = 0
    for i in range(1, n - 1):
        SS0 += X_Return[i] - X_Return[0]
    SS0 +=(X_Return[-1] - X_Return[0]) / 2
    SS0 = np.abs(SS0)

    S_ = 0
    for i in range(1, n - 1):
        S_ += (X_Return[i] - X_Return[0])-(x0[i]-x0[0])
    S_ += (X_Return[-1] - X_Return[0]-(x0[-1] - x0[0])) / 2
    S_ = np.abs(SS0)
    T = (1+S0+SS0)/(1+S0+SS0+S_)
    if T >= 0.9:
        print('精度为一级')
        print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)
    elif T >= 0.8:
        print('精度为二级')
        print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)
    elif T >= 0.7:
        print('精度为三级')
        print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)
    elif T >= 0.6:
        print('精度为四级')
        print('GM(1,1)模型为:\nX(k) = ', x0[0] - b / a, 'exp(', -a, '(i-1))', b / a)

if __name__ == '__main__':
    # 初始化原始数据
    history_data = [2.874, 3.278, 3.337, 3.390, 3.679]
    index = pd.period_range('2001', '2005', freq='A-DEC')
    GM_Model(history_data, index)

