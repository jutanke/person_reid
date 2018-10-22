import keras
from keras.models import Sequential, Model
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Dense, Dropout, Input, Flatten, concatenate
from os.path import isfile, isdir, join
from os import makedirs
from pak.util import download as dl
import cv2
import numpy as np
import h5py
import warnings


class ReId:

    def __init__(self, root='/tmp', url=None, name=None):
        """
            create a new instance of the ReId network
        :param root:
        """
        if url is None:
            url = 'http://188.138.127.15:81/models/reid.h5'
        if name is None:
            name = 'reid.h5'
        if not isdir(root):
            makedirs(root)

        filepath = join(root, name)
        if not isfile(filepath):
            print('could not find model.. downloading it')
            dl.download(url, filepath)

        if keras.__version__.startswith('2.2'):
            warnings.warn(
                "This model only works properly with keras 2.1.3. Weights for other versions might not work properly")

        # ------- build model -------
        seq = Sequential()
        xception = Xception(weights='imagenet', input_shape=(221, 221, 3),
                            include_top=False, pooling='avg')
        seq.add(xception)

        # freeze first layers in pre-trained model
        for layer in xception.layers[0:-20]:
            layer.trainable = False

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
        print('fp:', filepath)
        model.load_weights(filepath)
        self.model = model

    def predict(self, A, B):
        """
            compare two images
        :param A: images, range [0 .. 255]
        :param B:
        :return:
        """
        s1 = 221
        s2 = 221
        size = (s1, s2)
        if isinstance(A, list) or len(A.shape) == 4:
            assert len(A) == len(B)
            n = len(A)
            assert n > 0
            Xa = np.zeros((n, s1, s2, 3))
            Xb = np.zeros((n, s1, s2, 3))
            for idx, (a, b) in enumerate(zip(A, B)):
                Xa[idx, :, :, :] = cv2.resize(a, size)
                Xb[idx, :, :, :] = cv2.resize(b, size)
            Xa = preprocess_input(Xa)
            Xb = preprocess_input(Xb)
        elif len(A.shape) == 3:
            a = A
            b = B
            assert len(b.shape) == 3
            w1, h1, c1 = a.shape
            w2, h2, c2 = b.shape
            assert c1 == c2 == 3
            
            if w1 != s1 or h1 != s2:
                a = cv2.resize(a, size)
            if w2 != s1 or h2 != s2:
                b = cv2.resize(b, size)
            Xa = preprocess_input(a.astype('float64'))
            Xb = preprocess_input(b.astype('float64'))
            Xa = np.expand_dims(Xa, axis=0)
            Xb = np.expand_dims(Xb, axis=0)
        else:
            raise ValueError('wrong input shape' + str(A.shape))
        
        Y = self.model.predict([Xa, Xb])
        return Y[:, 0]
