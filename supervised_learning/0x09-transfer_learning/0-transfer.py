#!/usr/bin/env python3
""" 0x09. Transfer Learning """

import tensorflow as tf
import tensorflow.keras as K


# Load CIFAR10 Data
(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()


def preprocess_data(X, Y):
    """
        Script that trains a convolutional neural network
        to classify the CIFAR 10 dataset
    """

    # Keras Application expects a specific kind of input preprocessing
    X_p = K.applications.inception_resnet_v2.preprocess_input(X)

    # Add a file numbers from 0 to 9 categorically (CIFAR10 Categories)
    Y_p = K.utils.to_categorical(Y, 10)

    return (X_p, Y_p)


# if __name__ == '__main__':

# Apply preprocessing
X_train, Y_train = preprocess_data(X_train, Y_train)
X_test, Y_test = preprocess_data(X_test, Y_test)

# Transfer Learning Start

base_model = K.applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=(299, 299, 3))

# Start inputs with 32x32 size
inputs = K.Input(shape=(32, 32, 3))

# Lambda layer that scales up the data to the correct size
input = K.layers.Lambda(
    lambda image: tf.image.resize(
        image, (299, 299)))(inputs)

# Base Model Layers
x = base_model(input, training=False)
x = K.layers.GlobalAveragePooling2D()(x)
x = K.layers.Dense(500, activation='relu')(x)
x = K.layers.Dropout(0.3)(x)  # For Overfitting
outputs = K.layers.Dense(10, activation='softmax')(x)

# Mount the model
model = K.Model(inputs, outputs)

base_model.trainable = False  # Freeze
optimizer = K.optimizers.Adam()

# Adam, stochastic gradient descent method that
# is based on adaptive estimation of first-order
# and second-order moments.

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["acc"])  # Accuracy

history = model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_test,
        Y_test),
    batch_size=300,
    epochs=4,
    verbose=1)

# verbose = 1, which includes both progress bar and one line per epoch.
# verbose = 0, means silent.
# verbose = 2, one line per epoch i.e. epoch no./total no

model.save('cifar10.h5')
