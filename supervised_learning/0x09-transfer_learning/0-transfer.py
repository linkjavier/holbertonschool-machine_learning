#!/usr/bin/env python3
""" 0x09. Transfer Learning """


import tensorflow.keras as K


def preprocess_data(X, Y):
    """
        Script that trains a convolutional neural network
        to classify the CIFAR 10 dataset
    """

    X = X.astype('float32')
    StartX = K.applications.inception_v3.preprocess_input(X)
    StartY = K.utils.to_categorical(Y, 10)
    return (StartX, StartY)


if __name__ == '__main__':

    (CifarX, CifarY), (X, Y) = K.datasets.cifar10.load_data()
    PreCifarX, PreCifarY = preprocess_data(CifarX, CifarY)
    StartX, StartY = preprocess_data(X, Y)

    base = K.applications.InceptionV3(include_top=False, weights='imagenet')
    base.trainable = False
    model = K.Sequential()
    model.add(K.layers.Lambda(lambda x: K.backend.resize_images(x, 9, 9, 'channels_last', 'bilinear')))
    model.add(base)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []

    def rate_decay(epoch):
        decay = 0.001 / (1 + (0.01 * epoch))
        return (decay)

    learning = K.callbacks.LearningRateScheduler(schedule=rate_decay, verbose=1)
    callback.append(learning)
    callback.append(K.callbacks.ModelCheckpoint('cifar10.h5', monitor='val_accuracy', save_best_only=True, mode='max'))
    opt = K.optimizers.SGD(learning_rate=0.001) model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x=PreCifarX, y=PreCifarY, batch_size=128, epochs=30, verbose=1, shuffle=True, validation_data=(StartX, StartY), callbacks=callback)
