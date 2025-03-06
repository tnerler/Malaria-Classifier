from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.models import Sequential
import tensorflow as tf

def cnn_model(INPUT_SHAPE=(128, 128, 3)) : 

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size = 3, strides = 1, padding='same', activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    return model