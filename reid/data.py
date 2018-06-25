import numpy as np
from pak.datasets.CUHK03 import cuhk03
from pak.datasets.Market1501 import Market1501
from pak.datasets.DukeMTMC import DukeMTMC_reID
from pak.datasets.UMPM import UMPM
from os import makedirs
from os.path import join, isfile, isdir
import cv2
from numpy.random import randint
from random import random


def get_positive_pairs_by_index(Y):
    """ get all positive pairs in Y
    """
    n = len(Y)
    positive_pairs = []
    for i in range(n):
        if Y[i] > 0:
            for j in range(n):
                if Y[i] == Y[j]:
                    positive_pairs.append((i,j))
    return np.array(positive_pairs)


def get_bb(cam, pts3d):
    """ return (x, y, w, h)
    """
    assert len(pts3d) == 15
    rvec = np.array(cam['rvec']).astype('float32')
    tvec = np.array(cam['tvec']).astype('float32')
    K = cam['K'].astype('float32')
    distCoef = np.array(cam['distCoeff']).astype('float32')
    pts2d = np.squeeze(cv2.projectPoints(pts3d.astype('float32'),
                                         rvec,
                                         tvec, K, distCoef)[0])

    x_max = np.max(pts2d[:, 0])
    x_min = np.min(pts2d[:, 0])
    y_max = np.max(pts2d[:, 1])
    y_min = np.min(pts2d[:, 1])

    n1, n2, n3, n4 = randint(0, 50, 4)
    y = max(y_min - n1, 0)
    x = max(x_min - n2, 0)
    w = min(x_max - x + n3, 644)
    h = min(y_max - y + n4, 486)
    return int(x), int(y), int(w), int(h)


class UMPMSampler:

    def __init__(self, root,
                 umpm_datasets, umpm_user, umpm_password, w, h):
        """
        :param root: root folder for data
        :param umpm_datasets [{string}, ... ] defines which datasets are
            being used for
        :param umpm_user: must be set if len(umpm_datasets) > 0
        :param umpm_password: must be set if len(umpm_datasets) > 0
        """
        self.n = len(umpm_datasets)
        assert self.n > 0
        umpm = UMPM(root, umpm_user, umpm_password)
        self.cameras = ['l', 'r', 's', 'f']
        self.Xs = []
        self.Ys = []
        self.Calibs = []
        for ds in umpm_datasets:
            X, Y, Calib = umpm.get_data(ds)
            self.Xs.append(X)
            self.Ys.append(Y)
            self.Calibs.append(Calib)
        self.w = w
        self.h = h

    def get_random_sample(self, start, end, same_person):
        """
        :param start: {int} start frame
        :param end: {int} end frame
        :param same_person: {boolean}
        """
        person1 = randint(0, 2)
        if same_person:
            person2 = person1
        else:
            person2 = 0 if person1 == 1 else 1

        dataset = randint(0, self.n)  # choose dataset
        frame1, frame2 = randint(start, end, 2)  # frames
        cid1, cid2 = randint(0, 4, 2)  # camera
        cid1 = self.cameras[cid1]
        cid2 = self.cameras[cid2]
        cam1 = self.Calibs[dataset][cid1]
        cam2 = self.Calibs[dataset][cid2]

        X1 = self.Xs[dataset][cid1][frame1]
        X2 = self.Xs[dataset][cid2][frame2]

        start1 = person1 * 15
        pts3d_1 = self.Ys[dataset][frame1][start1:start1 + 15, 0:3]
        x1, y1, w1, h1 = get_bb(cam1, pts3d_1)

        start2 = person2 * 15
        pts3d_2 = self.Ys[dataset][frame2][start2:start2 + 15, 0:3]
        x2, y2, w2, h2 = get_bb(cam2, pts3d_2)

        im1 = X1[y1:y1 + h1, x1:x1 + w1]
        im2 = X2[y2:y2 + h2, x2:x2 + w2]
        return cv2.resize(im1, (self.w, self.h)), cv2.resize(im2, (self.w, self.h))

    def get_test(self, batch_size=16):
        """ take the first 100 frames as test set
        """
        start = 0
        end = 100
        X = []
        Y = []
        for i in range(batch_size):
            same_person = random() > 0.5
            im1, im2 = self.get_random_sample(start, end, same_person)
            X.append((im1, im2))
            Y.append([1, 0] if same_person else [0, 1])

        return np.array(X), np.array(Y)


class DataSampler:
    """ helps to sample person-ReId data from different sources
    """

    def __init__(self, root, target_w, target_h, cuhk03_test_T=100):
        """
        :param root: root folder for data
        :param target_w: {int} force all data to be of this w
        :param target_h: {int} force all data to be of this h
        :param cuhk03_test_T: {int} for the cuhk03 dataset: all ids > T are
            set as training data, the rest is test. This is not needed
            for Market and Duke as they come with their own train/test sets
        :param umpm_datasets [{string}, ... ] defines which datasets are
            being used for
        :param umpm_user: must be set if len(umpm_datasets) > 0
        :param umpm_password: must be set if len(umpm_datasets) > 0
        """
        self.root = join(root, "DataSampler")
        if not isdir(self.root):
            makedirs(self.root)

        cuhk = cuhk03(root, target_w=target_w, target_h=target_h)
        market = Market1501(root, force_shape=(target_w, target_h))
        duke = DukeMTMC_reID(root, force_shape=(target_w, target_h))

        # -- handle cuhk --
        self.handle_cuhk03(cuhk, cuhk03_test_T)

        # -- handle Market --
        self.market_X, self.market_Y, self.market_pos_pairs, \
            self.market_X_test, self.market_Y_test, self.market_pos_pairs_test = \
            self.handle(market, 'market')

        # -- handle Duke --
        self.duke_X, self.duke_Y, self.duke_pos_pairs, \
            self.duke_X_test, self.duke_Y_test, self.duke_pos_pairs_test = \
            self.handle(duke, 'duke')

    def handle(self, dataset, dataset_name):
        """ handle Market and Duke
        """
        X_test, Y_test = dataset.get_test()
        Y_test = Market1501.extract_ids(Y_test)
        X, Y = dataset.get_train()
        Y = Market1501.extract_ids(Y)

        fname_test = self.get_pos_pairs_file_name(dataset_name + '_test')
        if isfile(fname_test):
            pos_pairs_test = np.load(fname_test)
        else:
            pos_pairs_test = get_positive_pairs_by_index(Y_test)
            np.save(fname_test, pos_pairs_test)

        print("(" + dataset_name + ") positive test pairs: ", len(pos_pairs_test))

        fname_train = self.get_pos_pairs_file_name(dataset_name + '_train')
        if isfile(fname_train):
            pos_pairs_train = np.load(fname_train)
        else:
            pos_pairs_train = get_positive_pairs_by_index(Y)
            np.save(fname_train, pos_pairs_train)

        print("(" + dataset_name + ") positive train pairs: ", len(pos_pairs_train))

        return X, Y, pos_pairs_train, X_test, Y_test, pos_pairs_test

    def get_pos_pairs_file_name(self, dataset):
        """ gets the file name for the positive pairs
        """
        file_name = 'positive_pairs_' + dataset + '.npy'
        return join(self.root, file_name)

    def get_train_batch(self, num_pos, num_neg):
        """ gets a random batch from the test sets
        """
        pos_split, neg_split = int(num_pos/3), int(num_neg/3)
        X1, Y1 = DataSampler.sample_generic_batch(pos_split, neg_split,
            self.market_X, self.market_Y, self.market_pos_pairs)
        X2, Y2 = DataSampler.sample_generic_batch(pos_split, neg_split,
            self.duke_X, self.duke_Y, self.duke_pos_pairs)
        X3, Y3 = self.get_cuhk_train_batch(num_pos-2*pos_split,num_neg-2*neg_split)

        X = np.concatenate([X1, X2, X3])
        Y = np.concatenate([Y1, Y2, Y3])

        # scramble
        n = num_pos + num_neg
        order = np.random.choice(n, size=n, replace=False)

        return X[order], Y[order]

    def get_test_batch(self, num_pos, num_neg):
        """ gets a random batch from the test sets
        """
        assert num_pos > 2
        assert num_neg > 2
        num_pos_left, num_neg_left = num_pos, num_neg
        pos_split, neg_split = int(num_pos/3), int(num_neg/3)
        X1, Y1 = DataSampler.sample_generic_batch(pos_split, neg_split,
            self.market_X_test, self.market_Y_test, self.market_pos_pairs_test)
        X2, Y2 = DataSampler.sample_generic_batch(pos_split, neg_split,
            self.duke_X_test, self.duke_Y_test, self.duke_pos_pairs_test)
        X3, Y3 = self.get_cuhk_test_batch(num_pos-2*pos_split,num_neg-2*neg_split)

        X = np.concatenate([X1, X2, X3])
        Y = np.concatenate([Y1, Y2, Y3])

        # scramble
        n = num_pos + num_neg
        order = np.random.choice(n, size=n, replace=False)

        return X[order], Y[order]

    @staticmethod
    def sample_generic_batch(num_pos, num_neg, X, Y, pos_pairs):
        """ generic batch-sampling for Market and Duke. This function
            does not work on cuhk!
        """
        assert num_pos > 0
        assert num_neg > 0
        pos_indx = np.random.choice(len(pos_pairs), size=num_pos, replace=False)
        sampled_pos_pairs = pos_pairs[pos_indx]
        sampled_neg_pairs = []
        assert len(X) == len(Y)
        n = len(X)
        while len(sampled_neg_pairs) < num_neg:
            a, b = np.random.choice(n, size=2, replace=False)
            if Y[a] != Y[b]:
                sampled_neg_pairs.append((a,b))
        sampled_neg_pairs = np.array(sampled_neg_pairs)

        Ap = sampled_pos_pairs[:,0]
        Bp = sampled_pos_pairs[:,1]
        An = sampled_neg_pairs[:,0]
        Bn = sampled_neg_pairs[:,1]
        X_a_pos = X[Ap]
        X_b_pos = X[Bp]
        X_a_neg = X[An]
        X_b_neg = X[Bn]

        X_a = np.concatenate([X_a_pos, X_a_neg])
        X_b = np.concatenate([X_b_pos, X_b_neg])

        X = np.concatenate((X_a, X_b), axis=3)
        Y = np.array([(1, 0)] * num_pos + [(0, 1)] * num_neg)

        return X, Y

    # -----------------------------------------------
    # CUHK03
    # -----------------------------------------------

    def handle_cuhk03(self, cuhk, T):
        """ handle the cuhk data which has to be split into train/test set
            first
        """
        X, Y = cuhk.get_labeled()
        self.cuhk_X = X
        self.cuhk_Y = Y

        index_test, index_train = [], []
        for i, y in enumerate(Y):
            if y <= T:
                index_test.append(i)
            else:
                index_train.append(i)

        self.cuhk_index_test = np.array(index_test)
        self.cuhk_index_train = np.array(index_train)

        # test pairs
        fpairs_test = self.get_pos_pairs_file_name('cuhk_test')
        if isfile(fpairs_test):
            self.cuhk_test_pos_pair = np.load(fpairs_test)
        else:
            self.cuhk_test_pos_pair = []
            for i in index_test:
                for j in index_test:
                    if Y[i] == Y[j]:
                        self.cuhk_test_pos_pair.append((i, j))
            self.cuhk_test_pos_pair = np.array(self.cuhk_test_pos_pair)
            np.save(fpairs_test, self.cuhk_test_pos_pair)
        print("(cuhk) positive test pairs:", len(self.cuhk_test_pos_pair))

        # train pairs
        fpairs_train = self.get_pos_pairs_file_name('cuhk_train')
        if isfile(fpairs_train):
            self.cuhk_train_pos_pair = np.load(fpairs_train)
        else:
            self.cuhk_train_pos_pair = []
            for i in index_train:
                for j in index_train:
                    if Y[i] == Y[j]:
                        self.cuhk_train_pos_pair.append((i, j))
            self.cuhk_train_pos_pair = np.array(self.cuhk_train_pos_pair)
            np.save(fpairs_train, self.cuhk_train_pos_pair)
        print("(cuhk) positive train pairs:", len(self.cuhk_train_pos_pair))

    def get_cuhk_test_batch(self, num_pos, num_neg):
        return self.get_cuhk_batch(num_pos, num_neg,
                              self.cuhk_test_pos_pair, self.cuhk_index_test)

    def get_cuhk_train_batch(self, num_pos, num_neg):
        return self.get_cuhk_batch(num_pos, num_neg,
                              self.cuhk_train_pos_pair, self.cuhk_index_train)

    def get_cuhk_batch(self, num_pos, num_neg, pos_pairs, valid_indexes):
        """ generic batch function
        """
        pos_indx = np.random.choice(len(pos_pairs), size=num_pos, replace=False)
        sampled_pos_pairs = pos_pairs[pos_indx]
        sampled_neg_pairs = []
        Y = self.cuhk_Y
        X = self.cuhk_X

        n_all_indexes = len(valid_indexes)
        while len(sampled_neg_pairs) < num_neg:
            a, b = np.random.choice(
                n_all_indexes, size=2, replace=False)
            if Y[a] != Y[b]:
                sampled_neg_pairs.append((a,b))

        sampled_neg_pairs = np.array(sampled_neg_pairs)
        Ap = sampled_pos_pairs[:,0]
        Bp = sampled_pos_pairs[:,1]
        An = sampled_neg_pairs[:,0]
        Bn = sampled_neg_pairs[:,1]

        X_a_pos = X[Ap]
        X_b_pos = X[Bp]
        X_a_neg = X[An]
        X_b_neg = X[Bn]

        X_a = np.concatenate([X_a_pos, X_a_neg])
        X_b = np.concatenate([X_b_pos, X_b_neg])

        X = np.concatenate((X_a, X_b), axis=3)
        Y = np.array([(1, 0)] * num_pos + [(0, 1)] * num_neg)

        return X, Y