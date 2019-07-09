from model.DHDN import DHDN
from util.data_script import load_images,add_noise,get_data
from util.util import calculate_psnr
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
import os
from pathlib import Path
import tensorflow as tf
def learning_rate_schedule(epoch):
    return learning_rate/(2**(epoch//3))

def psnr_metrics(y_true,y_pred):
    # img1 and img2 have range [0, 1]
    return tf.image.psnr(y_true,y_pred,1)

# training configuration
batch_size = 8
max_epoches = 250
learning_rate = 0.0001
lr_drop = 20
os.environ["CUDA_VISIBLE_DEVICES"]="0"
noise_coefficient = 0.02
# set data_path
# windows
# data_path = 'E:\\zlh\\workspace\\python\\pro3_denoising\\DHDN_simulate\\dataset\\CBSD68'
# 225
# data_path = '/media/tclwh2/wangshupeng/zoulihua/DHDN_simulate/dataset/CBSD68'
# 11
data_path = '/home/tcl/zoulihua/DHDN_simulate/dataset/CBSD68'

load_model = True
load_model_path = '../experiment/model/model_001-0.05.hdf5'

# The data, shuffled and split between train and test sets:
y_train = load_images(data_path) # get_data(data_path,0.01)

# data augmentation
# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                    #  validation_split=0.1
                     )
image_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=noise_coefficient),**data_gen_args)
mask_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=0.),**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
# (std, mean, and principal components if ZCA whitening is applied).
seed = 5
# image_datagen.fit(x_train, augment=True, seed=seed)
# mask_datagen.fit(y_train, augment=True, seed=seed)

y_train = y_train/255
image_generator = image_datagen.flow(y_train,batch_size=batch_size,seed=seed)
mask_generator = mask_datagen.flow(y_train,batch_size=batch_size,seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

validation_x,validation_y = get_data(data_path,var=noise_coefficient)
validation_x,validation_y = validation_x/255,validation_y/255
# learning rate decrease
lrs = LearningRateScheduler(learning_rate_schedule, verbose=1)

# checkpoint save

model_prefix = "../experiment/model/"
Path().mkdir(exist_ok=True,parents=True)
model_path = model_prefix + "model_{epoch:03d}-{loss:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True, mode='auto')

# tensorboard
log_filepath = '../experiment/keras_log'
tb_cb = TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=0)

callbacks = [lrs, model_checkpoint,tb_cb]



# optimization details
adam = Adam(lr=learning_rate)
model = DHDN()
model.compile(loss=mean_absolute_error, optimizer=adam, metrics=['accuracy',psnr_metrics])


if load_model:
    assert Path(load_model_path).exists(),'can not load the model from the path, maybe is not exist'
    model.load_weights(load_model_path)

# training
# hostory_temp = model.fit(x_train,y_train,
#                         batch_size,max_epoches,
#                         verbose=1,
#                         callbacks=callbacks,
#                         validation_split=0.1,
#                         shuffle=True)

hostory_temp = model.fit_generator(train_generator,
                    steps_per_epoch=y_train.shape[0] // batch_size,
                    epochs=max_epoches,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(validation_x[-100:],validation_y[-100:])
                    )