from sklearn.model_selection import train_test_split


import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, save_model
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from time import sleep

from keras import backend as K

# x_train, y_train, x_test, y_test = [], [], [], []


frames_data = {}
frames_data["data"] = []

frames_data["target"] = []

num_classes = 2
batch_size = 64
epochs = 10


data = open("bigger-faces.csv", "r")

for i in data:
    emotion, pixel = i.split(",")
    pixels = pixel.split(" ")[0:9216]
    pixels = np.array(pixels, 'float32')
    frames_data['data'].append(pixels)
    emotion = keras.utils.to_categorical(emotion, num_classes)
    frames_data['target'].append(emotion)

x_train, x_test, y_train, y_test = train_test_split(
    frames_data["data"], frames_data["target"], test_size=0.2)


x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 96, 96, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 96, 96, 1)
x_test = x_test.astype('float32')


def train2():
    print("training...")
    model = Sequential()
    model.add(Convolution2D(32, (8, 8),
                            input_shape=(96, 96, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(2, 2)))
    K.set_image_dim_ordering('th')
    model.add(Convolution2D(64, 8, 8, activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(2, 2)))

    model.add(Convolution2D(128, 8, 8, activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(2, 2)))

    model.add(Convolution2D(64, 8, 8, activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit_generator(
        train_generator, steps_per_epoch=batch_size, epochs=epochs)

    # overall evaluation

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', 100*score[1])


def train3():
    print("Start training")
    model = Sequential()

    # 1st convolution layer
    model.add(Convolution2D(
        64, (5, 5), activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    #------------------------------

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    #------------------------------

    # model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
    # train for randomly selected one
    model.fit_generator(
        train_generator, steps_per_epoch=batch_size, epochs=epochs)

    # overall evaluation

    score = model.evaluate(x_test, y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', 100*score[1])

    # model.save("cnn_faces.h5")


train3()
