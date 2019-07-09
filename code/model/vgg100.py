from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Softmax
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

class cifar100vgg:
    def __init__(self,train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model_oct()
        self.model_path = 'cifar100vgg.h5'
        self.model_path = 'cifar100oct.h5'
        if train:
            self.model.load_weights('model_39-0.61.hdf5')
            self.model = self.train(self.model)
        else:
            self.model.load_weights(self.model_path)


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        inputs = Input(shape=(32,32,3,))

        # model = Sequential()
        weight_decay = self.weight_decay
        h1 = Conv2D(64, (3, 3), padding='same',input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.3)(h1)

        h1 = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = MaxPooling2D(pool_size=(2, 2))(h1)


        h1 = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)

        h1 = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = MaxPooling2D(pool_size=(2, 2))(h1)


        h1 = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)

        h1 = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)

        h1 = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = MaxPooling2D(pool_size=(2, 2))(h1)

        h1 = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)

        h1 = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)

        h1 = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = MaxPooling2D(pool_size=(2, 2))(h1)



        h1 = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)


        h1 = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.4)(h1)

        h1 = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = MaxPooling2D(pool_size=(2, 2))(h1)
        h1 = Dropout(0.5)(h1)

        h1 = Flatten()(h1)
        h1 = Dense(512,kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.5)(h1)
        
        h1 = Dense(self.num_classes)(h1)
        outputs = Softmax()(h1)

        model = Model(inputs=inputs,outputs=outputs)
        model.summary()
        return model

    def build_model_oct(self):
        from octave_conv_block import initial_oct_conv_bn_relu,final_oct_conv_bn_relu,oct_conv_bn_relu
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        inputs = Input(shape=(32,32,3,))
        # model = Sequential()
        weight_decay = self.weight_decay

        xh,xl = initial_oct_conv_bn_relu(inputs,64)
        xh, xl = Dropout(0.3)(xh), Dropout(0.3)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=64)
        xh, xl = MaxPooling2D()(xh), MaxPooling2D()(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=128)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=128)
        xh, xl = MaxPooling2D()(xh), MaxPooling2D()(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=256)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=256)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=256)
        xh, xl = MaxPooling2D()(xh), MaxPooling2D()(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=512)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=512)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=512)
        xh, xl = MaxPooling2D()(xh), MaxPooling2D()(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=512)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        xh, xl = oct_conv_bn_relu(xh, xl, filters=512)
        xh, xl = Dropout(0.4)(xh), Dropout(0.4)(xl)

        h1 = final_oct_conv_bn_relu(xh, xl, filters=512)
        h1 = MaxPooling2D()(h1)
        h1 = Dropout(0.5)(h1)

        h1 = Flatten()(h1)
        h1 = Dense(512,kernel_regularizer=regularizers.l2(weight_decay))(h1)
        h1 = ReLU()(h1)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.5)(h1)
        
        h1 = Dense(self.num_classes)(h1)
        outputs = Softmax()(h1)

        model = Model(inputs=inputs,outputs=outputs)
        model.summary()
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.05
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)


        

        # 设置 learning rate 减小的回调函数
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=10, min_lr=1e-6)

        # 设置 checkpoint 保存的回调函数
        model_path="model_{epoch:02d}-{val_acc:.2f}.hdf5"
        model_checkpoint = ModelCheckpoint(model_path, monitor="val_acc", save_best_only=True,
                                        save_weights_only=True, mode='auto')

        # 设置tensorboard 回调函数
        log_filepath = 'keras_log_oct'
        tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)

        callbacks = [lr_reducer, model_checkpoint,tb_cb]

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=callbacks,verbose=1)
        model.save_weights(self.model_path)
        return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)

    model = cifar100vgg(train=True)

    predicted_x = model.predict(x_test)
    residuals = (np.argmax(predicted_x,1)!=np.argmax(y_test,1))
    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
