import numpy as np
import keras
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist
np.random.seed(100)

classes = 10

# input image dimensions
img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if keras.backend.image_data_format() == 'channels_first':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, classes)

# Load pre-trained model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print (score[1]*100)
