# -- coding: UTF-8 --
import os, codecs
import shutil

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def get_file_name(path):
    '''''
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

        nhist = cv2.calcHist([img], [0,1], None, [64,64], [0.0, 255.0,0.0, 255.0])
        # img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        # nhist = cv2.calcHist([img], [0, 1, 2], None, [64, 64, 64],
        #                      [0, 256, 0, 256, 0, 256])
        nhist = cv2.normalize(nhist, nhist, 0, 255, cv2.NORM_MINMAX).flatten()

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.dtype)
        # Kp, des = sift.detectAndCompute(gray, None)
        # 检测并计算描述符
        # Kp,des=sift.detectAndCompute(gray,None)#检测并计算描述符
        # des =sift.detect(gray, None)# sift.detectAndCompute(gray, None)
        # 找到后可以计算关键点的描述符
        # Kp, des = sift.compute(gray, des)
        # if des is None:
        #     file_list.remove(file)
        #     continue
        # reshape_feature = img.reshape(-1, 1)
        # features.append(reshape_feature[0].tolist())
        features.append(nhist)

    input_x = np.array(features)
    # for k in range(4, 10):
    kmeans = KMeans(n_clusters=cluster_nums)
    kmeans.fit(input_x)
    value = sum(np.min(cdist(input_x, kmeans.cluster_centers_, 'euclidean'), axis=1)) / input_x.shape[0]
    # print(k, value)
    # a.append(value)

    # cha = [a[i] - a[i + 1] for i in range(len(a) - 1)]
    # a_v = a[cha.index(max(cha)) + 1]
    # index = a.index(a_v) + 1
    # print(max(cha), a_v, index)
    # kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x)

    return kmeans.labels_, kmeans.cluster_centers_


def res_fit(filenames, labels):
    files = [file.split('/')[-1] for file in filenames]

    for i in range(len(labels)):
        shutil.copy("/data/captcha/images/"+files[i], "./pic/"+str(labels[i])+files[i])
    return dict(zip(files, labels))


def save(path, filename, data):
    file = os.path.join(path, filename)
    with codecs.open(file, 'w', encoding='utf-8') as fw:
        for f, l in data.items():
            fw.write("{}\t{}\n".format(f, l))


def main():
    path_filenames = get_file_name("/data/captcha/images")

    print('开始读取数据')
    labels, cluster_centers = knn_detect(path_filenames,100)

    print ('开始填充数据')
    res_dict = res_fit(path_filenames, labels)

    print('开始保存数据')
    save('./', 'knn_res.txt', res_dict)


if __name__ == "__main__":
    main()
