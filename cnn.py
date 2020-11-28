from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import os
from joblib import load, dump
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

if os.path.exists('cnn.h5'):
    model = load_model('cnn.h5')
else:
    model = Sequential()

    model.add(Conv2D(10, (15, 15), activation='relu', kernel_regularizer=L1L2(l1=0.01, l2=0.01), input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.1, shuffle=True)

    model.save('cnn.h5')


