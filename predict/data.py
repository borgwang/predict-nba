# author: wangguibo <borgwang@126.com>
# date: 2017-8-23
# Copyright (c) 2017 by wangguibo
#
# file: data.py
# desc: Dataset class for predicting class
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from sklearn.model_selection import train_test_split
from utils.utils import *


class Data(object):

    def __init__(self):
        dataset_path = './dataset/dataset.p'
        dataset, filed_names = load_from_file(dataset_path, 'dataset')
        self.X = dataset[:, :-1]
        self.y = dataset[:, -1]
        self.filed_names = filed_names
        # split train/test set
        self.train_X, self.test_X, self.train_y, self.test_y = \
            train_test_split(self.X, self.y, train_size=0.9, random_state=0)
        self.train_size = len(self.test_X)

    def get_test_set(self):
        return self.test_X, self.test_y

    def get_train_set(self):
        return self.train_X, self.train_y

    @property
    def num_features(self):
        return len(self.filed_names)

    @property
    def target(self):
        return [0.0, 1.0]

    @property
    def num_outputs(self):
        return len(self.target)

    @property
    def feature_names(self):
        return self.filed_names
