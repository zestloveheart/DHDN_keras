import os
from pathlib import Path
from cv2 import cv2
import numpy as np
import logging

from util.data_script import get_data
from util.util import calculate_psnr,calculate_ssim,setup_logger

os.environ["CUDA_VISIBLE_DEVICES"]=""
save_image = True
visible_image = False
start, number = 85,10
# set path
# data_path = '..\\dataset\\CBSD68' # Windows
data_path = '../dataset/CBSD68' # Linux

# The data, shuffled and split between train and test sets:
x_train, y_train = get_data(data_path,0.02)
x_train, y_train = x_train/255, y_train/255


setup_logger('base', result_path, 'test', level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(f'Test {start} to {start+ number} Model : {load_model_path}')
sum_psnr = 0
for index,noise_img in enumerate(x_train[start:start+number]):
    temp_img = np.reshape(noise_img,(1,64,64,3))
    groundtruth_img = y_train[start+index]

    noise_img*=255
    denoise_img*=255
    groundtruth_img*=255

    logger.info(f"img : {start+index} {index}/{number}")
    logger.info(f"noise / groundtruth : psnr {calculate_psnr(noise_img,groundtruth_img)} ; ssim {calculate_ssim(noise_img,groundtruth_img)}")
    logger.info(f"denoise / groundtruth : psnr {calculate_psnr(denoise_img,groundtruth_img)} ; ssim {calculate_ssim(denoise_img,groundtruth_img)}")
    sum_psnr += calculate_psnr(denoise_img,groundtruth_img)

    if visible_image:
        cv2.imshow(f"{start+index}_noise.png",noise_img)
        cv2.imshow(f"{start+index}_denoise.png",denoise_img)
        cv2.imshow(f"{start+index}_groundtruth.png",groundtruth_img)
        cv2.waitKey(0)
logger.info(f"Average psnr : {sum_psnr/number}")