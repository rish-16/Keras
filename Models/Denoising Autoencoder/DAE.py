import numpy as np
import skimage
from keras.datasets import mnist
from keras.models import Model, model_from_json, Input
from keras.layers import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
np.random.seed(100)

# Loading MNIST characters into memory
(X_train, _), (_, _) = mnist.load_data()

X_train = X_train.reshape([60000, 784])
X_train = X_train.astype('float32')
X_train /= 255

noisy_X_train = []

# Adding noise to training images
for i in range(len(X_train)):
	try:
		noisy_image = skimage.util.random_noise(X_train[i], mode='gaussian', seed=None, clip=True)
		noisy_image = skimage.util.random_noise(noisy_image, mode='speckle', seed=None, clip=True)
		noisy_image = skimage.util.random_noise(noisy_image, mode='poisson', seed=None, clip=True)
		noisy_image = skimage.util.random_noise(noisy_image, mode='s&p', seed=None, clip=True)
		noisy_image = skimage.util.random_noise(noisy_image, mode='salt', seed=None, clip=True)
		noisy_image = skimage.util.random_noise(noisy_image, mode='pepper', seed=None, clip=True)
		noisy_X_train.append(noisy_image)
	except:
		pass

noisy_X_train = np.array(noisy_X_train, dtype=np.float32)
noisy_X_train = noisy_X_train.reshape([60000, 784])

# Building Autoencoder architecture
input_img = Input(shape=[784,])
hidden_1 = Dense(128, activation='relu')(input_img)
code = Dense(32, activation='relu')(hidden_1)
hidden_2 = Dense(128, activation='relu')(code)
output_img = Dense(784, activation='sigmoid')(hidden_2)

autoencoder = Model(input_img, output_img)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
autoencoder.summary()
autoencoder.fit(noisy_X_train, X_train, epochs=10)

# Saving model
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
autoencoder.save_weights("model.h5")
print("Saved model to disk")
