import numpy as np
from pak.datasets.CUHK03 import cuhk03
from pak.datasets.Market1501 import Market1501
from pak.datasets.DukeMTMC import DukeMTMC_reID
from pak.datasets.UMPM import UMPM
from pak.datasets.MOT import MOT16
from pak import utils
from os import makedirs
from os.path import join, isfile, isdir
import cv2
from numpy.random import randint
from random import random
from numpy.random import choice, uniform


def random_contrast_brightness(im):
    """
    """
    alpha = uniform(0.8, 1.1)
    beta = uniform(-10, 10)
    return np.clip(alpha * im + beta, 0, 255).astype('uint8')


class Data:
    """
    Samples data from all samplers
    """

    def __init__(self, root, target_w, target_h):
        """
        :param root:
        :param target_w:
        :param target_h:
        :return:
        """
        self.mot16 = MOT16Sampler(root, target_w, target_h)
        #self.mot16 = MOT16Sampler(root, target_w, target_h, video='MOT16-05')
        self.mot16 = [
            MOT16Sampler(root, target_w, target_h),
            MOT16Sampler(root, target_w, target_h, video='MOT16-05'),
            MOT16Sampler(root, target_w, target_h, video='MOT16-09'),
            MOT16Sampler(root, target_w, target_h, video='MOT16-10'),
            MOT16Sampler(root, target_w, target_h, video='MOT16-11'),
            MOT16Sampler(root, target_w, target_h, video='MOT16-13')
        ]

        self.mot16_test = MOT16Sampler(root, target_w, target_h, video='MOT16-04')
        self.reid = DataSampler(root, target_w, target_h)

    def train(self, batchsize=16, add_noise=True):
        """

        :param batchsize:
        :param add_noise: {boolean}
        :return:
        """
        bs1 = int(batchsize / 2)
        bs2 = batchsize - bs1
        bs2_pos = int(bs2/2)
        bs2_neg = bs2 - bs2_pos

        mot16 = self.mot16[randint(0, len(self.mot16))]

        x1, y1 = mot16.sample(bs1)
        x3, y3 = self.reid.get_train_batch(bs2_pos, bs2_neg)

        X = np.concatenate([x1, x3], axis=0)
        Y = np.concatenate([y1, y3], axis=0)

        if add_noise:
            size = np.prod(X.shape)
            noise = randint(-3, 3, size=size).reshape(X.shape)
            X = np.clip(X + noise, 0, 255)

        for idx in range(batchsize):
            X[idx, :, :, 0:3] = random_contrast_brightness(X[idx, :, :, 0:3])
            X[idx, :, :, 3:6] = random_contrast_brightness(X[idx, :, :, 3:6])
        
        n = batchsize
        order = np.random.choice(n, size=n, replace=False)
        return X[order], Y[order]

    def test(self, batchsize=16):
        """

        :param batchsize:
        :return:
        """
        bs1 = int(batchsize / 2)
        bs2 = batchsize - bs1
        bs2_pos = int(bs2 / 2)
        bs2_neg = bs2 - bs2_pos
        x1, y1 = self.mot16_test.sample(bs1)
        x2, y2 = self.reid.get_test_batch(bs2_pos, bs2_neg)
        X = np.concatenate([x1, x2], axis=0)
        Y = np.concatenate([y1, y2], axis=0)
        n = batchsize
        order = np.random.choice(n, size=n, replace=False)
        return X[order], Y[order]


# --- MOT16 ---
class MOT16Sampler:

    @staticmethod
    def get_visible_pedestrains(Y_gt):
        """ return people without distractors
        """
        Y_gt = utils.extract_eq(Y_gt, col=7, value=1)
        Y_gt = utils.extract_eq(Y_gt, col=8, value=1)
        return Y_gt

    def __init__(self, root, target_w, target_h, video="MOT16-02"):
        mot16 = MOT16(root)
        self.target_w = target_w
        self.target_h = target_h
        X, _, Y_gt = mot16.get_train(video, memmapped=True)
        Y_gt = MOT16Sampler.get_visible_pedestrains(Y_gt)  # only humans
        n_frames, h, w, _ = X.shape

        self.lookup = {}
        self.pid_frame_lookup = {}
        self.X = X
        self.n_frames = n_frames
        self.frames_with_persons = []
        self.pids_per_frame = {}

        for f in range(n_frames):
            Gt_per_frame = utils.extract_eq(Y_gt, col=0, value=f)
            n = len(Gt_per_frame)
            pids = Gt_per_frame[:, 1]
            left = Gt_per_frame[:, 2]
            top = Gt_per_frame[:, 3]
            width = Gt_per_frame[:, 4]
            height = Gt_per_frame[:, 5]
            valid_persons = 0
            valid_pids = []
            for pid, x, y, w, h in zip(*[pids, left, top, width, height]):
                # only take bbs that are 'big' enough
                if w > 50 and h > 100:
                    pid = int(pid)
                    if pid not in self.lookup:
                        self.lookup[pid] = []
                    self.lookup[pid].append(f)
                    H, W, _ = X[f].shape
                    x_left = max(0, int(x))
                    y_top = max(0, int(y))
                    x_right = min(W - 1, int(x + w))
                    y_bottom = min(H - 1, int(y + h))
                    self.pid_frame_lookup[pid, f] = \
                        (x_left, y_top, x_right, y_bottom)

                    valid_persons += 1
                    valid_pids.append(pid)

            self.pids_per_frame[f] = valid_pids

            if valid_persons > 1:
                self.frames_with_persons.append(f)

        del_pids = []  # (pid, frame) remove all pids who are only visible once
        for pid, visible_in_frames in self.lookup.items():
            assert len(visible_in_frames) > 0
            if len(visible_in_frames) == 1:
                del_pids.append((pid, visible_in_frames[0]))
        for pid, frame in del_pids:
            self.lookup.pop(pid)
            self.pid_frame_lookup.pop((pid, frame))
            valid_pids = self.pids_per_frame[frame]
            self.pids_per_frame[frame] = [p for p in valid_pids if p != pid]

        print('(MOT16) total number of bounding boxes:', len(self.pid_frame_lookup))

    def sample(self, batchsize=16):
        """ get a set of pairs
        """
        size = (self.target_w, self.target_h)
        pids = list(self.lookup.keys())
        X = np.zeros((batchsize, self.target_h, self.target_w, 6))
        Y = []
        for idx in range(batchsize):
            if random() > 0.5:  # get the same pair
                Y.append((1, 0))
                pid1 = pids[randint(0, len(pids))]
                pid2 = pid1
                possible_frames = self.lookup[pid1]
                assert len(possible_frames) > 1
                f1, f2 = choice(possible_frames, 2, replace=False)

            else:  # get a different pair
                Y.append((0, 1))
                f1, f2 = choice(self.frames_with_persons, 2)
                pid1, pid2 = -1, -1
                while pid1 == pid2:  # make sure we get two
                    # different persons
                    pid1 = choice(self.pids_per_frame[f1])
                    pid2 = choice(self.pids_per_frame[f2])

            im1 = cv2.resize(self.crop_bb(f1, pid1), size)
            im2 = cv2.resize(self.crop_bb(f2, pid2), size)

            X[idx, :, :, 0:3] = im1
            X[idx, :, :, 3:6] = im2

        return X, np.array(Y)

    def crop_bb(self, frame, pid, fizzle=5):
        """
        """
        fz1, fz2, fz3, fz4 = randint(-fizzle, fizzle, 4)
        im = self.X[frame]
        H, W, _ = im.shape
        res = self.pid_frame_lookup[pid, frame]
        x_left, y_top, x_right, y_bottom = res
        x_left = max(0, x_left + fz1)
        x_right = min(W - 1, x_right + fz2)
        y_top = max(0, y_top + fz3)
        y_bottom = min(H - 1, y_bottom + fz4)
        return im[y_top:y_bottom, x_left:x_right]


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


def get_bb(cam, pts3d, W=644, H=486, juggle=50):
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

    n1, n2, n3, n4 = randint(0, juggle, 4)
    y = max(y_min - n1, 0)
    x = max(x_min - n2, 0)
    w = min(x_max - x + n3, W)
    h = min(y_max - y + n4, H)
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
        if end < 0:
            end = len(self.Xs[dataset]['l'])
        assert 0 <= start < end
        assert end <= len(self.Xs[dataset]['l'])
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
        X = np.zeros((batch_size, self.h, self.w, 6))
        Y = []
        for i in range(batch_size):
            same_person = random() > 0.5
            im1, im2 = self.get_random_sample(start, end, same_person)
            X[i, :, :, 0:3] = im1
            X[i, :, :, 3:6] = im2
            Y.append([1, 0] if same_person else [0, 1])

        return X, np.array(Y)

    def get_train(self, batch_size=32):
        """

        :param batch_size:
        :return:
        """
        start = 100
        end = -1
        X = np.zeros((batch_size, self.h, self.w, 6))
        Y = []
        for i in range(batch_size):
            same_person = random() > 0.5
            im1, im2 = self.get_random_sample(start, end, same_person)
            X[i, :, :, 0:3] = im1
            X[i, :, :, 3:6] = im2
            Y.append([1, 0] if same_person else [0, 1])

        return X, np.array(Y)


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
