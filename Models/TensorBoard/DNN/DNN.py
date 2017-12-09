from time import time
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard
from keras.utils import np_utils
np.random.seed(100)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([x_train.shape[0], 784]).astype('float32')
x_test = x_test.reshape([x_test.shape[0], 784]).astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(500, input_shape=[784, ]))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), callbacks=[tensorboard], verbose=1)

# tensorboard --logdir=logs/
