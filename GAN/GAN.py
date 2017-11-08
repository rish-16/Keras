import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconv2D, Input
from keras.layers import LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle, random, sys
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
X_train /= 255
X_test /= 255

print (np.min(X_train), np.max(X_train))

print ('X_train shape: {}'.format(X_train.shape))
print ('Training samples: {}'.format(X_train.shape[0]))
print ('Test samples: {}'.format(X_test.shape[0]))

def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

shp = X_train[1:]

dropout = 0.25
gopt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)
nch = 200

shp = X_train.shape[1:]

dropout_rate = 0.25
# Optim

opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)
nch = 200

g_input = Input(shape=[100])
H = Dense(nch*14*14, kernel_initializer='glorot_normal')(g_input)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Reshape( [nch, 14, 14] )(H)
H = UpSampling2D(size=(2, 2))(H)
H = Conv2D(100, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Conv2D(25, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Conv2D(28, (1, 1), padding='same', kernel_initializer='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

d_input = Input(shape=shp)
H = Conv2D(256, (5, 5), strides=(2, 2), padding = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Conv2D(512, (5, 5), strides=(2, 2), padding = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2 ,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

make_trainable(discriminator, False)

gan_input = Input(shape=[100])
gan_V = discriminator(generator(gan_input))
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()

def plot_loss(losses):
	display.clear_output(wait=True)
	display.display(plt.gcf())
	plt.figure(figsize=(10,8))
	plt.plot(losses["d"], label='discriminitive loss')
	plt.plot(losses["g"], label='generative loss')
	plt.legend()
	plt.show()

def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
	noise = np.random.uniform(0,1,size=[n_ex,100])
	generated_images = generator.predict(noise)

	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
	    plt.subplot(dim[0],dim[1],i+1)
	    img = generated_images[i,0,:,:]
	    plt.imshow(img)
	    plt.axis('off')
	plt.tight_layout()
	plt.show()

ntrain = 10000
trainidx = random.sample(range(0,X_train.shape[0]), ntrain)
XT = X_train[trainidx,:,:,:]

noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
generated_images = generator.predict(noise_gen)
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
discriminator.fit(X,y, nb_epoch=1, batch_size=32)
y_hat = discriminator.predict(X)

y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print ("Accuracy: {} pct ({} of {}) right".format(acc, n_rig, n_tot))

losses = {"d":[], "g":[]}

def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):
	for e in tqdm(range(nb_epoch)):

        # Make generative images
		image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
		noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
		generated_images = generator.predict(noise_gen)

		# Train discriminator on generated images
		X = np.concatenate((image_batch, generated_images))
		y = np.zeros([2*BATCH_SIZE,2])
		y[0:BATCH_SIZE,1] = 1
		y[BATCH_SIZE:,0] = 1

		make_trainable(discriminator,True)
		d_loss  = discriminator.train_on_batch(X,y)
		losses["d"].append(d_loss)

		# train Generator-Discriminator stack on input noise to non-generated output class
		noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
		y2 = np.zeros([BATCH_SIZE,2])
		y2[:,1] = 1

		make_trainable(discriminator,False)
		g_loss = GAN.train_on_batch(noise_tr, y2 )
		losses["g"].append(g_loss)

        # Updates plots
		if e%plt_frq==plt_frq-1:
		    plot_loss(losses)
		    plot_gen()

train_for_n(nb_epoch=250, plt_frq=25,BATCH_SIZE=128)

K.set_value(opt.lr, 1e-4)
K.set_value(dopt.lr, 1e-5)
train_for_n(nb_epoch=100, plt_frq=10,BATCH_SIZE=128)


K.set_value(opt.lr, 1e-5)
K.set_value(dopt.lr, 1e-6)
train_for_n(nb_epoch=100, plt_frq=10,BATCH_SIZE=256)

plot_loss(losses)
plot_gen(25,(5,5),(12,12))

def plot_real(n_ex=16,dim=(4,4), figsize=(10,10) ):
	idx = np.random.randint(0,X_train.shape[0],n_ex)
	generated_images = X_train[idx,:,:,:]

	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = generated_images[i,0,:,:]
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()
	plt.show()

plot_real()
