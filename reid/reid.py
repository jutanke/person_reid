from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from os.path import isfile, isdir, join
from os import makedirs
from pak.util import download as dl
import cv2
import numpy as np


class ReId:

    def __init__(self, root='/tmp'):
        """
            create a new instance of the ReId network
        :param root:
        """
        url = 'http://188.138.127.15:81/models/stacknet64x64_84_BOTH.h5'
        name = 'stacknet64x64_84_BOTH.h5'
        if not isdir(root):
            makedirs(root)

        filepath = join(root, name)
        if not isfile(filepath):
            print('could not find model.. downloading it')
            dl.download(url, filepath)

        self.model = load_model(filepath)

    def predict(self, a, b):
        """
            compare two images
        :param a:
        :param b:
        :return:
        """
        w1, h1, c1 = a.shape
        w2, h2, c2 = b.shape
        assert c1 == c2 == 3
        s = 64
        if w1 != s or h1 != s:
            a = cv2.resize(a, (s, s))
        if w2 != s or h2 != s:
            b = cv2.resize(b, (s, s))

        a_max = np.max(a)
        b_max = np.max(b)
        if a_max > 1 or b_max > 1:
            # normalize to 1
            a = a.astype('float64')/255
            b = b.astype('float64')/255

        print('a', np.sum(a))
        print('b', np.sum(b))

        X = np.concatenate([a, b], axis=2)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X.astype('float64'))

        Y = self.model.predict(X)
        return np.squeeze(Y)[0]
