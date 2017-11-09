import numpy as np
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist
import matplotlib.pyplot as plt
np.random.seed(100)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)
autoencoder.load_weights("model.h5")
print("Loaded model from disk")

pred = autoencoder.predict(X_test)

fig = plt.figure()
for i in range(4):
	plt.subplot(2,4,i+1)
	plt.imshow(X_test[i].reshape([28,28]), cmap='gray')

for i in range(4):
	plt.subplot(2,4,i+5)
	plt.imshow(pred[i].reshape([28,28]), cmap='gray')

plt.show()
