import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf

# 数据文件夹
data_dir = "data"

def read_data(data_dir):
    datas = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        data = np.array(image) / 255.0
        datas.append(data)

    datas = np.array(datas)

    print("shape of datas: {}".format(datas.shape))
    return fpaths, datas

read_data(data_dir)