import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import Dense, Dropout, Input, Flatten, concatenate
from os.path import isfile, isdir, join
from os import makedirs
from pak.util import download as dl
import cv2
import numpy as np
import h5py


class ReId:

    def __init__(self, root='/tmp', url=None, name=None, verbose=False):
        """
            create a new instance of the ReId network
        :param root:
        """
        if url is None:
            url = 'http://188.138.127.15:81/models/model_heavy_89acc.h5'
        if name is None:
            name = 'model_heavy_89acc.h5'
        if not isdir(root):
            makedirs(root)

        filepath = join(root, name)
        if not isfile(filepath):
            print('could not find model.. downloading it')
            dl.download(url, filepath)

        if keras.__version__ == '2.1.3':
            self.model = load_model(filepath)
        else:
            # see bug https://github.com/jutanke/person_reid/issues/2
            with h5py.File(filepath, 'r') as f:
                model_weights = f.get('model_weights')
                layers = list(model_weights.keys())
                assert len(layers) == 8

                # --- build the model ---
                dense = DenseNet121(weights=None, input_shape=(221, 221, 3),
                                    include_top=False, pooling='avg')
                seq = Sequential()
                seq.add(dense)
                # seq.add(Flatten())  # not needed anymore

                input_a = Input(shape=(221, 221, 3))
                input_b = Input(shape=(221, 221, 3))

                out_a = seq(input_a)
                out_b = seq(input_b)

                concatenated = concatenate([out_a, out_b])
                hidden_drp1 = Dropout(0.5)(concatenated)
                hidden = Dense(32, activation='relu', name='hidden')(hidden_drp1)
                hidden_drp2 = Dropout(0.5)(hidden)
                out = Dense(1, activation='sigmoid', name='out')(hidden_drp2)

                model = Model([input_a, input_b], out)
                self.model = model
                # -----------------------
                # insert weights into model

                def insert_conv(model, key, conv_h5):
                    """ insert conv layer into model
                    :param model:
                    :param key:
                    :param conv_h5:
                    :return:
                    """
                    convl = model.get_layer(key)
                    kernel = conv_h5.get('kernel:0')[()]
                    convl.set_weights([kernel])

                def insert_bn(model, key, bn_h5):
                    """ insert the bn layer into the model
                    :param model:
                    :param key:
                    :param bn_h5:
                    :return:
                    """
                    beta = bn_h5.get('beta:0')[()]
                    gamma = bn_h5.get('gamma:0')[()]
                    mm = bn_h5.get('moving_mean:0')[()]
                    mv = bn_h5.get('moving_variance:0')[()]
                    bn_layer = model.get_layer(key)
                    bn_layer.set_weights([beta, gamma, mm, mv])

                # start with densenet
                _dense = model_weights.get('sequential_1')  # dense
                n = len(list(_dense.keys()))
                for i, k in enumerate(list(_dense.keys())):
                    if verbose and (i % 10) == 0:
                        print('\tload layer %03d/%03d' % (i+1, n))

                    if k == 'conv1':  # this one has changed...
                        conv = _dense.get(k).get('conv')
                        insert_conv(dense, 'conv1/conv', conv)
                        bn = _dense.get(k).get('bn')
                        insert_bn(dense, 'conv1/bn', bn)
                    else:
                        if k.endswith('bn'):
                            bn = _dense.get(k)
                            insert_bn(dense, k, bn)
                        elif k.endswith('conv'):
                            conv = _dense.get(k)
                            insert_conv(dense, k, conv)
                        else:
                            raise Exception('invalid layer name:' + str(k))

                # care for the fc layers now
                fc1 = model_weights.get('dense_1').get('dense_1')
                bias = fc1.get('bias:0')[()]
                kernel = fc1.get('kernel:0')[()]
                hidden_layer = model.get_layer('hidden')
                hidden_layer.set_weights([kernel, bias])

                fc2 = model_weights.get('dense_2').get('dense_2')
                bias = fc2.get('bias:0')[()]
                kernel = fc2.get('kernel:0')[()]
                out_layer = model.get_layer('out')
                out_layer.set_weights([kernel, bias])

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
