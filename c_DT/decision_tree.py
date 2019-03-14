#!/usr/bin/env python
# encoding: utf-8
"""
@Author     :xinbei
@Software   :Pycharm
@File       :decision_tree.py
@Time       :2019/3/13 8:56
@Version    :
@Purpose    :
"""
import math
import operator


# 生成简单数据集
def create_dataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算香农熵 l(x) = -log2 p(x)
def calc_shannon_ent(dataSet):
    """
    :param dataSet: data set matrix
    :return:
    """
    """
    num_entries: rows number of data set 
    label_counts: count each class of label(string to int)
    shannoent：entropy of shannoon
    """
    num_entries = len(dataSet)
    label_counts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannoent = 0.0
    # 以二为底求对数
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries # 计算概率，p(x)
        shannoent -= prob * math.log(prob, 2)
    return shannoent

# 划分数据集
def split_dataset(dataSet, axis, value):
    """
    :param dataSet:  data set matrix
    :param axis: feature used to split data set
    :param value:expected value of feature
    :return: new matrix which is satisfied requirement but without required axis
    """
    """
    retdataSet: expected data set matrix
    """
    # 创建新的list对象
    retdataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retdataSet.append(reducedFeatVec)

    return retdataSet


# 选择最好的特征划分数据集
def choose_best(dataSet):
    """
    :param dataSet:  data set matrix
    :return:
    """
    """
    numFeatures: number of features
    baseEntropy: entropy of complete data set
    bestInfoGain: best information
    bestFeature: best feature(most entropy)
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calc_shannon_ent(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    # 创建唯一分类标签
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValis = set(featList) # 混乱顺序的list
        newEntropy = 0.0

        # 计划每种划分的信息墒
        for value in uniqueValis:
            subDataSet = split_dataset(dataSet, i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calc_shannon_ent(subDataSet)
            infoGain = baseEntropy - newEntropy

            # 计算最好的增益墒
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i

    return bestFeature


# 统计出现次数并返回字典的函数
def majoritycnt(classList):
    """
    :param classList:
    :return: reverse sorted dict of class and their appearance
    """
    """
    classCount: number of class labels appearance
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
            sortedClassCount = sorted(classCount.items(), \
                                      key=operator.itemgetter(1), reverse=True)

            return sortedClassCount

if __name__ == '__main__':
    pass