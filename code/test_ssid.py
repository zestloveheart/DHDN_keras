import os
from pathlib import Path
from cv2 import cv2
import numpy as np
import logging

from model.model_util import model_getter
from util.data_script import get_data,divide_image,merge_image
from util.util import calculate_psnr,calculate_ssim,setup_logger
from util.ssid_script import load_benchmark
os.environ["CUDA_VISIBLE_DEVICES"]="0"
save_image = True
visible_image = False
start, number = 0,68
# set path
data_path = '../dataset/SIDD_Benchmark_Data' # Linux
model_name = "DHDN"
load_model_path = '../experiment/model/DHDN_015-0.02.hdf5'
result_path = '../experiment/result/'

# The data, shuffled and split between train and test sets:
x_train = load_benchmark(data_path)/255

model = model_getter(model_name,load_model_path)

Path(result_path).mkdir(exist_ok=True,parents=True)
setup_logger('base', result_path, 'test', level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(f'Test {start} to {start+ number} Model : {load_model_path}')
for index,noise_img in enumerate(x_train[start:start+number]):
    temp_img = divide_image(noise_img,64)
    denoise_imgs = model.predict(temp_img)*255
    denoise_img = merge_image(denoise_imgs,noise_img,64)
    noise_img*=255
    
    # whole calcu
    logger.info(f"img : {start+index} {index}/{number}")

    if save_image:
        cv2.imwrite(f"{result_path}{start+index}_noise.png",noise_img)
        cv2.imwrite(f"{result_path}{start+index}_denoise.png",denoise_img)
    if visible_image:
        cv2.imshow(f"{start+index}_noise.png",noise_img)
        cv2.imshow(f"{start+index}_denoise.png",denoise_img)
        cv2.waitKey(0)