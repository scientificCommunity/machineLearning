# -- coding: UTF-8 --
from PIL import Image
# coding=utf-8
from numpy import *
import os
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


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
    myCentroids, clustAssing = kMeans(dataMat, 3)
    show(dataMat, 3, myCentroids, clustAssing)
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
    features = zeros([len(imlist), 7500])  # 特征长度512
    for i, f in enumerate(imlist):
        im = Image.open(f)  # Image不是image包，是PIL里的Image模块
        # multi-dimensional histogram
        im = im.resize((50, 50))
        im = array(im).flatten()
        features[i] = im.flatten()
    # data = np.loadtxt('K-means_data')
    return features


if __name__ == '__main__':
    main()