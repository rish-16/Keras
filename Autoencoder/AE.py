import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
np.random.seed(100)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

input_img = Input(shape=[784,])
hidden_1 = Dense(128, activation='relu')(input_img)
code = Dense(32, activation='relu')(hidden_1)
hidden_2 = Dense(128, activation='relu')(code)
output_img = Dense(784, activation='sigmoid')(hidden_2)

autoencoder = Model(input_img, output_img)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
autoencoder.summary()
autoencoder.fit(X_train, X_train, epochs=5)

model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
autoencoder.save_weights("model.h5")
print("Saved model to disk")
