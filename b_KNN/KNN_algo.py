#!/usr/bin/env python
# encoding: utf-8
"""
@Author     :xinbei
@Software   :Pycharm
@File       :KNN_algo.py
@Time       :2019/3/10 18:04
@Version    :
@Purpose    :knn algorithm coding
"""
## 导入模块
import numpy as np
import operator
from os import listdir


## 生成数据
def createDataSet():
    """
    :return: X and Y data set
    """
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
group, labels = createDataSet()


## knn分类器算法,投票制
def classify0(inX, dataSet, labelsSet, k):
    """
    :param inX: test point(to make sure its label)
    :param dataSet: training data set
    :param labelsSet: labels vector of data set
    :param k:number of nearest neighbors
    :return: high percentage of labels around
    """
    """
    Parameters:
    -----------------------------------------
    dataSetSize: rows number of data set matrix
    sqDiffMat: square of difference between 'inX' and data set
    sqInX: count square of 'inX'
    distances:  Euclidean distance from each point to point[0,0]
    sortedDistIndicies: list 'distances' sorting rank(start from 0)
    classCount: 
        voteIlabel:label of top three distance, one by one
        classCount:count labels of top three distance
    sortedClassCount: rank the numbers of labels from more to less
    """
    dataSetSize = dataSet.shape[0]
    sqDiffMat = (np.tile(inX, (dataSetSize, 1)) - dataSet) ** 2
    distances = sqDiffMat.sum(axis=1) ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labelsSet[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
print('OK:',classify0([0,0], group, labels, 3))

## 文本数据转为数据集、标签集
def file2matrix(filename):
    """
    :param filename: file
    :return returnMat: a new matrix to store data set(no labels)
            classLabelVector: labels set(list)
    """
    """
    Parameters:
    -----------------------------------------
    love_dictionary: quantifying degree of preference(dict)
    arrayOLines: file content
    numberOfLines: number of file lines(samples numbers)
    returnMat: a new matrix to store data set(no labels)
        listFromLine: three arguments and one label in each sample (list)
        classLabelVector: labels set(list)
    """
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    with open(filename) as fr:
        arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
# dataSetMat, labelsSetList = file2matrix('datingTestSet.txt')


## 归一化数据: newValue = (oldValue-min) / (max-mi)
def autoNorm(dataSet):
    """
    :param dataSet: data set
    :return: normalized data set, difference, minimum
    """
    """
    Parameters:
    -----------------------------------------
    minVals: maximum in data set
    maxVals: minimum in data set
    ranges: difference between 'maxVals' and 'minVals'
    normDataSet: a new matrix after data set normalized
    """
    minVals, maxVals = dataSet.min(0), dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = np.zeros(np.shape(dataSet))
    # m = dataSet.shape[0]
    # normDataSet = dataSet - np.tile(minVals, (m, 1))
    # normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide

    normDataSet = (dataSet-minVals)/(maxVals-minVals)
    return normDataSet, ranges, minVals



## 测试分类器效果
def datingClassTest():
    """
    :return: self knn algorithm error
    """
    """
    Parameters:
    -----------------------------------------
    hoRatio: 
    datingDataMat: data set
    datingLabels: labels set
    normMat, ranges, minVals: normalized data set, difference, minimum
    """
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix(r'./data/datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:]
                                     ,datingLabels[numTestVecs:m],3)
        print("the classifier came back with: {}, the real answer is: {}"\
              .format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is:{}".format(errorCount/float(numTestVecs)))
    print(errorCount)



# 约会网站识别
def classifyPerson():
    """
    :return: None
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix(r'./data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3) # 归一化并调用knn

    print("You will probably like this person: {}".format(resultList[classifierResult - 1]))



# 转换图像txt为向量
def img2vector(filename):
    """
    :param filename: image file path or name
    :return:
    """
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect



# 手写数字识别
def handwritingClassTest():
    """
    :return:
    """
    """
    trainingFileList: a list contains all file names
    m: number of files
    trainingMat: matrix contains all training set
    classNumStr: label of each sample in file 
    """
    hwLabels = []
    # load the training set
    trainingFileList = listdir('./data/trainingDigits')

    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('data/trainingDigits/{}'\
                                      .format(fileNameStr))

    testFileList = listdir('data/testDigits') # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {}, the real answer is: {}"\
              .format(classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %{}".format(errorCount))
    print("\nthe total error rate is: {}".format(errorCount/float(mTest)))

if __name__ == '__main__':
    pass