from keras.models import load_model
from keras.applications.densenet import preprocess_input
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
        url = 'http://188.138.127.15:81/models/model_heavy_89acc.h5'
        name = 'model_heavy_89acc.h5'
        if not isdir(root):
            makedirs(root)

        filepath = join(root, name)
        if not isfile(filepath):
            print('could not find model.. downloading it')
            dl.download(url, filepath)

        self.model = load_model(filepath)

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
