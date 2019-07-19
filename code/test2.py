from keras.preprocessing.image import ImageDataGenerator
import os
from pathlib import Path
import tensorflow as tf

from util.data_script import load_images,add_noise,patch_generator
import time
# training configuration
batch_size = 8
max_epoches = 250
learning_rate = 0.0001
os.environ["CUDA_VISIBLE_DEVICES"]="0"
noise_coefficient = 0.02
model_name = "DHDN"

# set data_path
# data_path = '..\\dataset\\CBSD68' # Windows
data_path = '../dataset/CBSD68' # Linux


import psutil





# # set model_path
# load_model_path = '../experiment/model/DHDN_015-0.02.hdf5'

# if not Path(data_path+"_patch").exists():
#     patch_generator(data_path)
# data_path+="_patch"

# # load data

# # data augmentation
# # we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                    #  validation_split=0.1
                     )

test_datagen = ImageDataGenerator(**data_gen_args)
for i in test_datagen.flow_from_directory(data_path,target_size = (64,64),batch_size=8,class_mode=None):
    print(i.shape)
    info = psutil.virtual_memory()
    print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：',info.total)
    print(u'内存占比：',info.percent)
    print(u'cpu个数：',psutil.cpu_count())
    time.sleep(1)
    
# image_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=noise_coefficient),**data_gen_args)
GT_datagen = ImageDataGenerator(**data_gen_args)
# # Provide the same seed and keyword arguments to the fit and flow methods
# # (std, mean, and principal components if ZCA whitening is applied).
seed = 5
# # image_datagen.fit(x_train, augment=True, seed=seed)
# GT_datagen.fit(y_train, augment=True, seed=seed)
# image_generator = image_datagen.flow(y_train,batch_size=batch_size,seed=seed)
for i in GT_datagen.flow(y_train,batch_size=batch_size,seed=seed):
    print(i.shape)

    time.sleep(1)
# # combine generators into one which yields image and GT
# train_generator = zip(image_generator, GT_generator)
# # validation data
# # validation_x,validation_y = get_data(data_path,var=noise_coefficient)
# # validation_x,validation_y = validation_x/255,validation_y/255
