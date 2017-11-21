from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
images = []

for img_path in ['dog.jpg', 'elephant.jpeg', 'gorilla.jpeg', 'lion.jpeg', 'panda.jpeg', 'tiger.jpeg']:
	img = image.load_img(img_path, target_size=[224,224])
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	images.append(x)

for image in images:
	preds = model.predict(image)
	print ('Predicted: {}'.format(decode_predictions(preds, top=1)[0]))
