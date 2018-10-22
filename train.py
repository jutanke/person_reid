import keras
import tensorflow as tf
print('keras:\t', keras.__version__)
print('tf:\t', tf.__version__)

import json
from pprint import pprint
Settings = json.load(open('settings.txt'))
pprint(Settings)
print('')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from reid.data import DataSampler

from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.models import load_model
from os.path import join, isfile, isdir, exists, splitext
from keras.applications.xception import preprocess_input

root = Settings['data_root']

target_w = 221
target_h = 221

from reid.data import Data
import numpy as np


sampler = Data(root, target_w, target_h)
print('------------')


x, y = sampler.train()


def generate_training():
    global sampler
    while True:
        X, Y = sampler.train(add_noise=True)
        X_a = preprocess_input(X[:,:,:,0:3])
        X_b = preprocess_input(X[:,:,:,3:6])
        yield ([X_a, X_b], Y[:, 0])

def generate_test():
    global sampler
    while True:
        X, Y = sampler.test()
        X_a = preprocess_input(X[:,:,:,0:3])
        X_b = preprocess_input(X[:,:,:,3:6])
        yield ([X_a, X_b], Y[:, 0])


# ------------------------
gen = generate_training()

from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import load_model

filepath = join(root, 'reid.h5')
print('model:', filepath)

print('constructing...')
seq = Sequential()
xception = Xception(weights='imagenet', input_shape=(221, 221, 3),
    include_top=False, pooling='avg')
seq.add(xception)


# freeze first layers in pre-trained model
for layer in xception.layers[0:-20]:
    layer.trainable = False

#seq.add(Flatten())

input_a = Input(shape=(221, 221, 3))
input_b = Input(shape=(221, 221, 3))

out_a = seq(input_a)
out_b = seq(input_b)

concatenated = concatenate([out_a, out_b])
hidden1 = Dense(128, activation='relu', name='dense_1')(concatenated)
hidden_drp1 = Dropout(0.7)(hidden1)
hidden2 = Dense(32, activation='relu', name='dense_2')(hidden_drp1)
hidden_drp2 = Dropout(0.1)(hidden2)
out = Dense(1, activation='sigmoid', name='dense_3')(hidden_drp2)

model = Model([input_a, input_b], out)

if isfile(filepath):
    print('weights found... loading...')
    model.load_weights(filepath)

from keras.optimizers import SGD, RMSprop, Nadam, Adam, Adadelta

optimizer = Nadam()
#optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Nadam(lr=0.0001)
loss = 'binary_crossentropy'
metrics = ['binary_accuracy', 'acc']

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)


model.summary()

from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard

logs_dir = "./logs"
tb = TensorBoard(log_dir=logs_dir,
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_weights_only=True,
                             save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint, TerminateOnNaN(), tb]

history = model.fit_generator(generate_training(),
                             validation_data=generate_test(),
                             validation_steps=50,
                             steps_per_epoch=100,
                             epochs=2000,
                             callbacks=callbacks_list)

acc = history.history['val_binary_accuracy']
tacc = history.history['binary_accuracy']

plt.plot(range(len(acc)), acc)
plt.plot(range(len(tacc)), tacc)
plt.show()
