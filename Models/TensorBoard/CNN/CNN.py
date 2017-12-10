from time import time
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import TensorBoard
np.random.seed(100)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape([x_train.shape[0], 28, 28, 1]).astype('float32')
x_test = x_test.reshape([x_test.shape[0], 28, 28, 1]).astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=[28,28,1], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, callbacks=[tensorboard], validation_data=(x_test, y_test))

# tensorboard --logdir=logs/
