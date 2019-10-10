#!/usr/bin/python
# coding=utf-8
'''
基于直方图特征的图片聚类实现
'''
import numpy as np
import os
from PIL import Image
# coding=utf-8
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


# 计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


# 随机生成初始的质心（ng的课说的初始方式是随机选K个点）
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(np.array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


def main():
    dataMat = aa()
    myCentroids, clustAssing = kMeans(dataMat, 2)
    print dataMat
    show(dataMat, 2, myCentroids, clustAssing)

    print(clustAssing)
    path = '/data/captcha/images'
    imlist = os.listdir(path)
    for i, name in enumerate(imlist):
        print(path + name)
        im = Image.open(path + name)
        im.save('./img/{}/{}.jpg'.format(int(clustAssing[i, 0]), name))


def aa():
    path = '/data/captcha/images'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    # extract feature vector (8 bins per color channel)
    features = zeros([len(imlist), 512])  # 特征长度512
    for i, f in enumerate(imlist):
        im = array(Image.open(f))  # Image不是image包，是PIL里的Image模块
        # multi-dimensional histogram
        h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True,
                               range=[(0, 255), (0, 255), (0, 255)])  # reshape函数要导入Numpy 用其进行平整，将图像拉成一个三维数据。
        # print(len(h))=8
        # print(edges)=三个数组，每个数组8个数值，一共24个数据
        features[i] = h.flatten()
    path = '/data/captcha/images'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    # extract feature vector (8 bins per color channel)
    features = zeros([len(imlist), 512])  # 特征长度512
    for i, f in enumerate(imlist):
        im = array(Image.open(f))  # Image不是image包，是PIL里的Image模块
        # multi-dimensional histogram
        h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True,
                               range=[(0, 255), (0, 255), (0, 255)])  # reshape函数要导入Numpy 用其进行平整，将图像拉成一个三维数据。
        # print(len(h))=8
        # print(edges)=三个数组，每个数组8个数值，一共24个数据
        features[i] = h.flatten()
    # data = np.loadtxt('K-means_data')
    return features


if __name__ == '__main__':
    main()
