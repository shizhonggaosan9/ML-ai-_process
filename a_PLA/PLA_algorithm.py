#!/usr/bin/env python
# encoding: utf-8
"""
@Author     :xinbei
@Software   :Pycharm
@File       :02PLA_algorithm.py
@Time       :2019/1/21 14:42
@Version    :
@Purpose    :写带学习率的PLA算法以及Pocket PLA算法
"""

import numpy as np

"""PLA"""
# PLA主要算法
def PLA(X, Y):
    halt = 0  # 权重更新次数
    accuracy = 0  # 准确率得分
    rate = round(np.random.random() / 2.5 * 0.75 + 0.1, 2)  # 随机学习率0.1-0.4
    W = np.zeros(X.shape[1])

    # 遍历X每一行，即遍历每一组数据第一个元素：行数
    ### 每一组数据都能提升权重准确率，因此用遍历
    for i in range(X.shape[0]):  # shape属性第一个元素：行数
        a_result = np.dot(W, X[i, :])  # 计算当前权重所得结果
        if np.sign(a_result) != Y[i]:
            W = W + rate * np.dot(Y[i], X[i, :])  # 更新权重
            halt += 1  # 权重修改+1

    # 计算准确率得分
    ### 不需要提升权重准确率，不需要遍历
    g = np.sign(np.dot(X, W))
    accuracy = np.mean(g == Y)

    return halt, accuracy
    pass

def PLA_algo1(X, Y):
    # 初始版本PLA
    halt, accuracy = PLA(X, Y)

    return halt, accuracy
    pass

def PLA_algo2(X, Y):
    Iteration = 2000
    Halt = []
    Accuracy = []

    # 多次迭代利用随机种子生成不同数据
    for itera in range(Iteration):
        # 生成随机种子
        np.random.seed(itera)

        # 构建组合
        permutation = np.random.permutation(X.shape[0])
        X, Y = X[permutation], Y[permutation]

        halt, accuracy = PLA(X, Y)

        # 记录多次PLA各自的权重更新次数以及各自的得分
        Halt.append(halt)
        Accuracy.append(accuracy)

    # 计算多次迭代PLA后的次数均值以及得分均值
    halt_mean = np.mean(Halt)
    acc_mean = np.mean(Accuracy)

    return halt_mean, acc_mean
    pass

"""pocket PLA"""
# 计算权重W错误次数
def calError(X, Y, W):
    score = np.dot(X, W)
    Y_pred = np.sign(score)  # 将內积格式化
    err_cnt = np.sum(Y_pred != Y)
    return err_cnt

def pocket_PLA(X, Y):
    halt = 0  # 权重更新次数
    accuracy = 0  # 准确率得分
    rate = round(np.random.random() / 2.5 * 0.75 + 0.1, 2)  # 随机学习率0.1-0.4
    W = np.zeros(X.shape[1]) # 更优权重
    tmp = W # 当前权重
    count = 0 # 权重从更优权重更换到当前权重的次数
    min_err = 400 # 最小错误数量,初始值为总体数量
    tmp_err = 400 # 当前错误数量,初始值为总体数量
    Update = 100  # 搜索并计算错误的数据样本量，总体为400

    # 遍历X每一行，即遍历每一组数据第一个元素：行数
    ### 每一组数据都能提升权重准确率，因此用遍历
    for i in range(Update):  # 遍历Update组数据
        a_result = np.dot(W, X[i, :])  # 计算当前权重所得结果
        if np.sign(a_result) != Y[i]:
            tmp = W + rate * np.dot(Y[i], X[i, :])  # 更新当前权重
            tmp_err = calError(X, Y, tmp)
            halt += 1  # 权重修改+1
            if tmp_err<min_err:
                W = tmp
                min_err = tmp_err
                count += 1

    # 计算准确率得分
    ### 不需要提升权重准确率，不需要遍历
    g = np.sign(np.dot(X, W))
    accuracy = np.mean(g == Y)

    return halt, count, accuracy

def pocket_PLA_algo1(X, Y):
    Iteration = 2000
    Accuracy = [] # 统计错误次数

    for itera in range(Iteration):
        # 随机种子
        np.random.seed(itera)

        # 组合
        permutation = np.random.permutation(X.shape[0])
        X, Y = X[permutation], Y[permutation]

        # pocket PLA
        halt, count, accuracy = pocket_PLA(X, Y)

        Accuracy.append(accuracy)

    return halt, count, accuracy
    pass
if __name__ == '__main__':
    # easy test
    pass;