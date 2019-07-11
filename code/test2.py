import keras
import numpy
from keras.layers import Conv2D
import os

mnist = keras.datasets.mnist
(train_data,train_label),(test_data,test_label) = mnist.load_data()
train_data,test_data = train_data/255.0,test_data/255.0

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def create_model():
    inputs = keras.layers.Input(shape=(None,None,3,))
    x = keras.layers.Conv2D(10,(3,3),padding="same")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs= inputs,outputs = outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",metrics=['accuracy',])
    model.summary()
    return model

model = create_model()
#baseline_history = model.fit(train_data, train_label,
#                                 epochs=5,
#                                 validation_data=(test_data,test_label),
#                                 )
test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)
