from util.data_script import load_images,add_noise
from util.util import calculate_psnr
from keras.preprocessing.image import ImageDataGenerator
import os
from pathlib import Path
from cv2 import cv2
# training configuration
batch_size = 8
max_epoches = 250
learning_rate = 0.0001
lr_drop = 20
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# set data_path
# windows
data_path = 'E:\\zlh\\workspace\\python\\pro3_denoising\\DHDN_simulate\\dataset\\CBSD68'
# 225
# data_path = '/media/tclwh2/wangshupeng/zoulihua/DHDN_simulate/dataset/CBSD68'
# 11
# data_path = '/home/tcl/zoulihua/DHDN_simulate/dataset/CBSD68'

load_model = True
load_model_path = '../experiment/model/model_014-0.04.hdf5'

# The data, shuffled and split between train and test sets:
y_train = load_images(data_path) # get_data(data_path,0.01)

# data augmentation
# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                    #  validation_split=0.1
                     )
image_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=0.02),**data_gen_args)
mask_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=0.01),**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
# (std, mean, and principal components if ZCA whitening is applied).
seed = 5
# image_datagen.fit(x_train, augment=True, seed=seed)
# mask_datagen.fit(y_train, augment=True, seed=seed)

y_train = y_train/255
image_generator = image_datagen.flow(y_train,batch_size=batch_size,seed=seed)
mask_generator = mask_datagen.flow(y_train,batch_size=batch_size,seed=seed)
import numpy as np
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
print(train_generator.__next__()[0].dtype)
for batch in train_generator:
    x,y = batch
    for index in range(batch_size):
        print(np.mean(x[index]))
        print(np.mean(y[index]))
        cv2.imshow("noise",x[index])
        cv2.imshow("GT",y[index])
        cv2.waitKey(0)
