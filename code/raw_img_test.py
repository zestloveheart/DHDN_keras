import scipy.io as scio
from cv2 import cv2


import h5py

import rawpy
import imageio
import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
# 元信息读取
# meta_path='raw_rgb/0200_METADATA_RAW_010.MAT'
# data3 = scio.loadmat(meta_path)
# x3=list(data3.keys())
# print(len(data3['metadata'][0][0]))
# for _ in data3['metadata'][0][0]:
#     print(_)

# 单张图片读取
# raw_path='raw_rgb/0200_GT_RAW_010.mat'
# data=h5py.File(raw_path)
# data1 = data['x'][:1000,:1000]
# data_temp = None
# data1 = (data1*255).astype(np.uint8)
# li = [cv2.COLOR_BayerBG2BGR, cv2.COLOR_BayerGB2BGR, cv2.COLOR_BayerRG2BGR, cv2.COLOR_BayerGR2BGR, cv2.COLOR_BayerBG2RGB, cv2.COLOR_BayerGB2RGB, cv2.COLOR_BayerRG2RGB, cv2.COLOR_BayerGR2RGB]
# for i in li:
#     data_temp = cv2.cvtColor(data1, i, data_temp)
#     cv2.imshow("asdf",data_temp)
#     cv2.waitKey(0)

def classify_pattern(filename):
    device_pattern = {'IP':'rggb','S6':'grbg'}
    for i in device_pattern.keys():
        if i in filename:
            return device_pattern[i]

def bayer_unify_crop(img,pattern=None):
    # origin to BGGR for train
    # BGGR to origin for test as disunify
    if pattern == 'grbg':
        return img[1:-1,:]
    elif pattern == 'gbrg':
        return img[:,1:-1]
    elif pattern == 'rggb':
        return img[1:-1,1:-1]
    else:
        return img

def bayer_unify_pad(img,pattern=None):
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    # origin to BGGR for test
    if pattern == 'grbg':
        return np.pad(img, ((1, 1), (0, 0)), 'constant', constant_values=0)
    elif pattern == 'gbrg':
        return np.pad(img, ((0, 0), (1, 1)), 'constant', constant_values=0)
    elif pattern == 'rggb':
        return np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0)
    else:
        return img

def bayer_aug(img,mode):
    if mode == 'hor':
        return cv2.flip(img,1)[:,1:-1]
    elif mode == 'ver':
        return img[::-1][1:-1,:]
    elif mode == 'trans':
        return img.T
    return img
    
def pack_raw(raw):
    # pack Bayer image to 4 channels
    # im = raw.raw_image_visible.astype(np.float32)
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    raw_shape = raw.shape
    H = raw_shape[0]
    W = raw_shape[1]

    out = np.concatenate((raw[0:H:2, 0:W:2],
                          raw[0:H:2, 1:W:2],
                          raw[1:H:2, 1:W:2],
                          raw[1:H:2, 0:W:2]), axis=2)
    return out

def get_patch(img,size=64,step=None):
    patchs = []
    step = size if not step else step
    for i in range(0,img.shape[0],step):
        for j in range(0,img.shape[1],step):
            if i+size < img.shape[0] and j+size < img.shape[1]:
                patchs.append(img[i:i+size,j:j+size])
    return patchs

for filename in Path(".").rglob("*GT*.MAT"):
    raw_path=str(filename)
    data=h5py.File(raw_path)
    data1 = data['x'][500:1000,500:1000]
    # data1 = bayer_unify_crop(data1,classify_pattern(raw_path))
    pattern = classify_pattern(raw_path)
    data1 = bayer_unify_pad(data1,pattern)
    data1 = get_patch(data1)[0]
    # data1 = bayer_aug(data1,'hor')
    data_temp = None
    data1 = (data1*255).astype(np.uint8)
    # test different model
    # li = [cv2.COLOR_BayerBG2BGR, cv2.COLOR_BayerGB2BGR, cv2.COLOR_BayerRG2BGR, cv2.COLOR_BayerGR2BGR, cv2.COLOR_BayerBG2RGB, cv2.COLOR_BayerGB2RGB, cv2.COLOR_BayerRG2RGB, cv2.COLOR_BayerGR2RGB]
    # for i in li:
    #     data_temp = cv2.cvtColor(data1, i, data_temp)
    #     cv2.imshow("asdf",data_temp)
    #     cv2.waitKey(0)
    data_temp = cv2.cvtColor(data1, cv2.COLOR_BayerBG2RGB)
    cv2.imshow(raw_path,data_temp)
    cv2.waitKey(0)





# data2 = cv2.imread("raw_rgb/0200_GT_RAW_010.MAT_sRGB.png")
# print(data1.shape,data2.shape)







# with rawpy.imread('raw_rgb/0200_GT_RAW_010.mat') as raw:

#     #直接调用postprocess可能出现偏色问题
#     rgb = raw.postprocess()

#     #以下两行可能解决偏色问题，output_bps=16表示输出是 16 bit (2^16=65536)需要转换一次
#     #im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#     #rgb = np.float32(im / 65535.0*255.0)
#     #rgb = np.asarray(rgb,np.uint8)
#     imageio.imshow("asfas",rgb)