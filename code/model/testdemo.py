import ..
import utils
import keras
import numpy
from keras.layers import Conv2D



mnist = keras.datasets.mnist
(train_data,train_label),(test_data,test_label) = mnist.load_data()
train_data,test_data = train_data/255.0,test_data/255.0

# (train_data,train_label) = load_data_img()

def create_model():
    inputs = keras.layers.Input(shape=(None,None,3,))
    x = keras.layers.Conv2D(8,3,padding="same")(inputs)
    x = keras.layers.Conv2D(16,1,padding="same")(x)
    x = keras.layers.Conv2D(16,3,padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(16,3,strides=(2,2),padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(32,1,padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(32,3,padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(32,3,strides=(2,2),padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(64,3,padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(64,3,strides=(2,2),padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU()(x)
    x = keras.layers.Conv2D(20,1,padding="same")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Softmax(x)(x)

    model = keras.Model(inputs= inputs,outputs = outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",metrics=['accuracy',])
    model.summary()
    return model

create_model()

def create_model_cnn():
    inputs = keras.layers.Input(shape=(60,30,3,))

    h1 = keras.layers.Conv2D(8, kernel_size=(3, 3),padding="same", activation=keras.activations.relu)(inputs)
    h1 = keras.layers.Conv2D(16, kernel_size=(1, 1),padding="same", activation=keras.activations.relu)(h1)
    h1 = keras.layers.Conv2D(32, kernel_size=(3, 3),padding="same", activation=keras.activations.relu)(h1)
    h1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(h1)
    h1 = keras.layers.Conv2D(64, kernel_size=(1, 1), padding="same",activation=keras.activations.relu)(h1)
    h1 = keras.layers.Conv2D(64, kernel_size=(3, 3),padding="same", activation=keras.activations.relu)(h1)
    h1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(h1)
    h1 = keras.layers.Conv2D(64, kernel_size=(1, 1), padding="same",activation=keras.activations.relu)(h1)
    h1 = keras.layers.Conv2D(64, kernel_size=(3, 3),padding="same", activation=keras.activations.relu)(h1)
    h1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(h1)
    # h1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(h1)
    h1 = keras.layers.Flatten()(h1)

    h1 = keras.layers.Dense(1000, activation=keras.activations.relu)(h1)
    # h1 = keras.layers.Dropout(0.5)(h1)

    # h1 = keras.layers.Dense(1000,activation=tf.nn.relu)(h1)
    # h1 = keras.layers.Dropout(0.5)(h1)

    h1 = keras.layers.Dense(64, activation=keras.activations.relu)(h1)

    outputs = keras.layers.Dense(20, activation=keras.activations.relu)(h1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy", metrics=['accuracy', ])
    model.summary()
    return model

def create_model_cnn2():
    inputs = keras.layers.Input(shape=(28, 28,))
    h1 = keras.layers.Reshape((28,28,1))(inputs)
    h1 = keras.layers.Conv2D(32, kernel_size=(3, 3),strides=(1,1), activation=keras.activations.relu)(h1)
    h1 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation=keras.activations.relu)(h1)
    h1 = keras.layers.Dropout(0.5)(h1)
    h1 = keras.layers.Conv2D(64, kernel_size=(3, 3),strides=(1, 1),activation=keras.activations.relu)(h1)
    h1 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2),activation=keras.activations.relu)(h1)
    h1 = keras.layers.Dropout(0.5)(h1)
    h1 = keras.layers.Conv2D(10, 1, padding="same")(h1)
    h1 = keras.layers.GlobalAveragePooling2D()(h1)
    outputs = keras.layers.Softmax(h1)(h1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy", metrics=['accuracy', ])
    model.summary()
    return model
model = create_model_cnn2()
# model = create_model()

checkpoint_path = "single_digital2.ckpt"
import os
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    log_filepath = 'E:/TensorBoard/keras_log'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    baseline_history = model.fit(train_data, train_label,
                                 epochs=5,
                                 validation_data=(test_data,test_label),
                                 )

test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)

import matplotlib.pyplot as plt
for choose in range(len(test_label)):
    p = model.predict(numpy.array([test_data[choose]]),batch_size=1)
    if str(p.argmax())!=test_label[choose] and p.argmax()!=test_label[choose]:
        print(p.argmax(), test_label[choose])
        plt.imshow(test_data[choose])
        plt.show()