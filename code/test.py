import os
from pathlib import Path
from cv2 import cv2
import numpy as np
import logging

from model.DHDN import DHDN
from util.data_script import get_data
from util.util import calculate_psnr,calculate_ssim,setup_logger
os.environ["CUDA_VISIBLE_DEVICES"]="4"
save_image = True
visible_image = False
# set data_path
# windows
# data_path = 'E:\\zlh\\workspace\\python\\pro3_denoising\\DHDN_keras\\dataset\\CBSD68'
# 225
# data_path = '/media/tclwh2/wangshupeng/zoulihua/DHDN_keras/dataset/CBSD68'
# 11
data_path = '/home/tcl/zoulihua/DHDN_keras/dataset/CBSD68'

load_model_path = '../experiment/model/model_002-0.09.hdf5'
result_path = '../result/'
# The data, shuffled and split between train and test sets:
x_train, y_train = get_data(data_path,0.02)
x_train, y_train = x_train/255, y_train/255

model = DHDN()

assert Path(load_model_path).exists(),'can not load the model from the path, maybe is not exist'
model.load_weights(load_model_path)

setup_logger('base', result_path, 'test', level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')


start, number = 85,10
logger.info(f'Test {start} to {start+ number} Model : {load_model_path}')
sum_psnr = 0
for index,noise_img in enumerate(x_train[start:start+number]):
    temp_img = np.reshape(noise_img,(1,64,64,3))
    denoise_img = model.predict(temp_img)
    denoise_img = np.reshape(denoise_img,(64,64,3))
    groundtruth_img = y_train[start+index]

    noise_img*=255
    denoise_img*=255
    groundtruth_img*=255

    logger.info(f"img : {start+index} {index}/{number}")
    logger.info(f"noise / groundtruth : psnr {calculate_psnr(noise_img,groundtruth_img)} ; ssim {calculate_ssim(noise_img,groundtruth_img)}")
    logger.info(f"denoise / groundtruth : psnr {calculate_psnr(denoise_img,groundtruth_img)} ; ssim {calculate_ssim(denoise_img,groundtruth_img)}")
    sum_psnr += calculate_psnr(denoise_img,groundtruth_img)

    if save_image:
        cv2.imwrite(f"{result_path}{start+index}_noise.png",noise_img)
        cv2.imwrite(f"{result_path}{start+index}_denoise.png",denoise_img)
        cv2.imwrite(f"{result_path}{start+index}_groundtruth.png",groundtruth_img)
    if visible_image:
        cv2.imshow(f"{start+index}_noise.png",noise_img)
        cv2.imshow(f"{start+index}_denoise.png",denoise_img)
        cv2.imshow(f"{start+index}_groundtruth.png",groundtruth_img)
        cv2.waitKey(0)
logger.info(f"Average psnr : {sum_psnr/number}")