from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
import os
from pathlib import Path
import tensorflow as tf

from model.model_util import model_getter
from util.RgbDataLoader import RgbDataLoader
from util.util import calculate_psnr

def learning_rate_schedule(epoch):
    return learning_rate/(2**(epoch//3))

def psnr_metrics(y_true,y_pred):
    # img1 and img2 have range [0, 1]
    return tf.image.psnr(y_true,y_pred,1)

# training configuration
batch_size = 12
max_epoches = 250
learning_rate = 0.0001
os.environ["CUDA_VISIBLE_DEVICES"]="0"
noise_coefficient = 0.02
model_name = "DHDN"

# set data_path
# data_path = '..\\dataset\\CBSD68' # Windows
data_path = '../dataset/CBSD68' # Linux

# set model_path
load_model_path = '' # '../experiment/model/DHDN_015-0.02.hdf5'

# callbacks
# learning rate decrease
lrs = LearningRateScheduler(learning_rate_schedule, verbose=1)
# checkpoint save
model_prefix = "../experiment/model/"
Path(model_prefix).mkdir(exist_ok=True,parents=True)
model_path = model_prefix + model_name +"_{epoch:03d}-{loss:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(model_path,monitor='loss',save_best_only=True, save_weights_only=True, mode='auto')
# tensorboard
log_filepath = '../experiment/keras_log'
tb_cb = TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=0)
callbacks = [lrs, model_checkpoint,tb_cb]

# create model
# optimization details
adam = Adam(lr=learning_rate)
model = model_getter(model_name,load_model_path)
model.compile(loss=mean_absolute_error, optimizer=adam, metrics=['accuracy',psnr_metrics])

mdl = RgbDataLoader(data_path,batch_size,target_size=64,noise_coefficient=noise_coefficient)

# training
# hostory_temp = model.fit(x_train,y_train,
#                         batch_size,max_epoches,
#                         verbose=1,
#                         callbacks=callbacks,
#                         validation_split=0.1,
#                         shuffle=True)

hostory_temp = model.fit_generator(mdl.load_images_train(),
                    steps_per_epoch=len(mdl) // batch_size,
                    epochs=max_epoches,
                    verbose=1,
                    callbacks=callbacks,
                    )