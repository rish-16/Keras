import numpy as np
import skimage
from keras.models import model_from_json
from keras.datasets import mnist
import matplotlib.pyplot as plt
np.random.seed(100)

# Loading MNIST characters into memory
(_, _), (X_test, _) = mnist.load_data()

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

noisy_X_test = []

# Adding noise to training images
for i in range(len(X_test)):
	noisy_image = skimage.util.random_noise(X_test[i], mode='gaussian', seed=None, clip=True)
	noisy_X_test.append(noisy_image)

noisy_X_test = np.array(noisy_X_test, dtype=np.float32)
noisy_X_test = noisy_X_test.reshape([len(X_test), 784])

# Loading model into memory
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)
autoencoder.load_weights("model.h5")
print("Loaded model from disk")

preds = autoencoder.predict(noisy_X_test, batch_size=128)

fig = plt.figure()
for i in range(4):
	plt.subplot(2,4,i+1)
	plt.imshow(noisy_X_test[i].reshape([28,28]), cmap='gray')

for i in range(4):
	plt.subplot(2,4,i+5)
	plt.imshow(preds[i].reshape([28,28]), cmap='gray')

plt.show()
