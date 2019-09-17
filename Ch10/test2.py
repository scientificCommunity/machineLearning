# -*- encoding:utf-8 -*-
__date__ = '17/04/21'
'''
CV_INTER_NN - 最近邻插值,  
CV_INTER_LINEAR - 双线性插值 (缺省使用)  
CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法..  
CV_INTER_CUBIC - 立方插值
'''

import os, codecs
import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_file_name(path):
    '''
    Args: path to list;  Returns: path with filenames
    '''
    filenames = os.listdir(path)
    path_filenames = []
    filename_list = []
    for file in filenames:
        if not file.startswith('.'):
            path_filenames.append(os.path.join(path, file))
            filename_list.append(file)

    return path_filenames


def knn_detect(file_list, cluster_nums, randomState=None):
    features = []
    files = file_list
    sift = cv2.xfeatures2d.SIFT_create()
    for file in files:
        print(file)
        img = cv2.imread(file)
        data = np.array(img) / 255.0
        img = cv2.resize(img, (66, 67), interpolation=cv2.INTER_CUBIC)

        # 转换成灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.dtype)

        # 找特征点
        _, des = sift.detectAndCompute(gray, None)

        if des is None:
            file_list.remove(file)
            continue

        # 拉伸成一维向量
        reshape_feature = des.reshape(-1, 1)
        features.append(reshape_feature[0].tolist())

    input_x = np.array(features)

    kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x)

    return kmeans.labels_, kmeans.cluster_centers_


def res_fit(filenames, labels):
    files = [file.split('/')[-1] for file in filenames]

    return dict(zip(files, labels))


def save(path, filename, data):
    file = os.path.join(path, filename)
    with codecs.open(file, 'w', encoding='utf-8') as fw:
        for f, l in data.items():
            fw.write("{}\t{}\n".format(f, l))


def main(cluster_nums=2):
    path_filenames = get_file_name("./data/")

    labels, cluster_centers = knn_detect(path_filenames, cluster_nums)

    res_dict = res_fit(path_filenames, labels)
    save('./', 'knn_res.txt', res_dict)


if __name__ == "__main__":
    main()