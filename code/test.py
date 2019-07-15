import os
from pathlib import Path
from cv2 import cv2
import numpy as np
import logging

from model.model_util import model_getter
from util.data_script import get_data,divide_image,merge_image
from util.util import calculate_psnr,calculate_ssim,setup_logger

os.environ["CUDA_VISIBLE_DEVICES"]="0"
save_image = True
visible_image = False
start, number = 0,68
# set path
# data_path = '..\\dataset\\CBSD68' # Windows
data_path = '../dataset/CBSD68' # Linux
model_name = "DHDN"
load_model_path = '../experiment/model/DHDN_015-0.02.hdf5'
result_path = '../experiment/result/'

# The data, shuffled and split between train and test sets:
x_train, y_train = get_data(data_path,0.02)
x_train, y_train = x_train/255, y_train/255

model = model_getter(model_name,load_model_path)

Path(result_path).mkdir(exist_ok=True,parents=True)
setup_logger('base', result_path, 'test', level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(f'Test {start} to {start+ number} Model : {load_model_path}')
noise_psnr,denoise_psnr,noise_ssim,denoise_ssim = 0,0,0,0
for index,noise_img in enumerate(x_train[start:start+number]):
    temp_img = divide_image(noise_img,64)
    denoise_imgs = model.predict(temp_img)*255
    denoise_img = merge_image(denoise_imgs,noise_img,64)
    groundtruth_img = y_train[start+index]*255
    groundtruth_imgs = divide_image(groundtruth_img,64)
    noise_img*=255
    
    # patch calcu
    # temp_denoise_psnr,temp_denoise_ssim = 0,0
    # for index_patch in range(len(temp_img)):
    #     temp_denoise_psnr += calculate_psnr(denoise_imgs[index_patch],groundtruth_imgs[index_patch])
    #     temp_denoise_ssim += calculate_ssim(denoise_imgs[index_patch],groundtruth_imgs[index_patch])
    # temp_denoise_psnr,temp_denoise_ssim = temp_denoise_psnr/len(temp_img),temp_denoise_ssim/len(temp_img)
    # logger.info(f"img : {start+index} {index}/{number}")
    # logger.info(f"noise / groundtruth : psnr/ssim {calculate_psnr(noise_img,groundtruth_img)}/{calculate_ssim(noise_img,groundtruth_img)}")
    # logger.info(f"denoise / groundtruth : psnr/ssim {temp_denoise_psnr}/{temp_denoise_ssim}")
    # noise_psnr += calculate_psnr(noise_img,groundtruth_img)
    # noise_ssim += calculate_ssim(noise_img,groundtruth_img)
    # denoise_psnr += temp_denoise_psnr
    # denoise_ssim += temp_denoise_ssim

    # whole calcu
    logger.info(f"img : {start+index} {index}/{number}")
    logger.info(f"noise / groundtruth : psnr/ssim {calculate_psnr(noise_img,groundtruth_img)}/{calculate_ssim(noise_img,groundtruth_img)}")
    logger.info(f"denoise / groundtruth : psnr/ssim {calculate_psnr(denoise_img,groundtruth_img)}/{calculate_ssim(denoise_img,groundtruth_img)}")
    noise_psnr += calculate_psnr(noise_img,groundtruth_img)
    noise_ssim += calculate_ssim(noise_img,groundtruth_img)
    denoise_psnr += calculate_psnr(denoise_img,groundtruth_img)
    denoise_ssim += calculate_ssim(denoise_img,groundtruth_img)


    if save_image:
        cv2.imwrite(f"{result_path}{start+index}_noise.png",noise_img)
        cv2.imwrite(f"{result_path}{start+index}_denoise.png",denoise_img)
        cv2.imwrite(f"{result_path}{start+index}_groundtruth.png",groundtruth_img)
    if visible_image:
        cv2.imshow(f"{start+index}_noise.png",noise_img)
        cv2.imshow(f"{start+index}_denoise.png",denoise_img)
        cv2.imshow(f"{start+index}_groundtruth.png",groundtruth_img)
        cv2.waitKey(0)
logger.info(f"Average psnr/ssim : from {noise_psnr/number}/{noise_ssim/number} to {denoise_psnr/number}/{denoise_ssim/number} ")