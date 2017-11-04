import numpy as np
import keras
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist
np.random.seed(100)

# Load data and preprocess
classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

y_test = np_utils.to_categorical(y_test, classes)

# Load pre-trained model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Test pre-trained model
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print (score[1]*100)
