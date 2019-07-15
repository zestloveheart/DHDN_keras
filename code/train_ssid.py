from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
import os
from pathlib import Path
import tensorflow as tf

from model.model_util import model_getter
from util.ssid_script import load_images
from util.util import calculate_psnr
from cv2 import cv2
def learning_rate_schedule(epoch):
    return learning_rate/(2**(epoch//3))

def psnr_metrics(y_true,y_pred):
    # img1 and img2 have range [0, 1]
    return tf.image.psnr(y_true,y_pred,1)

# training configuration
batch_size = 8
max_epoches = 15
learning_rate = 0.0001
os.environ["CUDA_VISIBLE_DEVICES"]="0"
model_name = "DHDN"

# set data_path
# data_path = '..\\dataset\\CBSD68' # Windows
data_path = '../dataset/SIDD_Medium_Srgb/Data' # Linux

# set model_path
load_model_path = '../experiment/model/DHDN_015-0.02.hdf5'

# load data
x_train,y_train = load_images(data_path,5)
x_train,y_train = x_train/255, y_train/255
# data augmentation
# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     )
image_datagen = ImageDataGenerator(**data_gen_args)
GT_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
# (std, mean, and principal components if ZCA whitening is applied).
seed = 5
# image_datagen.fit(x_train, augment=True, seed=seed)
# GT_datagen.fit(y_train, augment=True, seed=seed)
image_generator = image_datagen.flow(x_train,batch_size=batch_size,seed=seed)
GT_generator = GT_datagen.flow(y_train,batch_size=batch_size,seed=seed)
# combine generators into one which yields image and GT
train_generator = zip(image_generator, GT_generator)

# validation data
# validation_x,validation_y = get_data(data_path,var=noise_coefficient)
# validation_x,validation_y = validation_x/255,validation_y/255

# callbacks
# learning rate decrease
lrs = LearningRateScheduler(learning_rate_schedule, verbose=1)
# checkpoint save
model_prefix = "../experiment/model/"
Path(model_prefix).mkdir(exist_ok=True,parents=True)
model_path = model_prefix + model_name +"_{epoch:03d}-{loss:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True, mode='auto')
# tensorboard
log_filepath = '../experiment/keras_log'
tb_cb = TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=0)
callbacks = [lrs, model_checkpoint,tb_cb]

# create model
# optimization details
adam = Adam(lr=learning_rate)
model = model_getter(model_name,load_model_path)
model.compile(loss=mean_absolute_error, optimizer=adam, metrics=['accuracy',psnr_metrics])

# training
# hostory_temp = model.fit(x_train,y_train,
#                         batch_size,max_epoches,
#                         verbose=1,
#                         callbacks=callbacks,
#                         validation_split=0.1,
#                         shuffle=True)

# hostory_temp = model.fit_generator(train_generator,
#                     steps_per_epoch=y_train.shape[0] // batch_size,
#                     epochs=max_epoches,
#                     verbose=1,
#                     callbacks=callbacks,
#                     )