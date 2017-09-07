# author: wangguibo <borgwang@126.com>
# date: 2017-8-28
# Copyright (c) 2017 by wangguibo
#
# file: main.py
# desc: Predict game result using ML methods.
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import runtime_path
import numpy as np
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from data import Data
from nn import NeuralNet


np.random.seed(666)


def cv_report(results, top_k=3):
    for i in range(1, top_k + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: %d' % i)
            print('Mean validation score: %.3f (std: %.3f)' % (
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print('Parameters: {0}\n'.format(results['params'][candidate]))


def svm_clf():
    print('SVM classifier...')
    clf = svm.SVC(random_state=0, tol=1e-4, max_iter=50000, kernel='rbf')
    param_dist = {
        'gamma': uniform(0.001, 10),
        'kernel': ['linear', 'rbf'],
        'C': uniform(0.1, 20)}
    return clf, param_dist


def rf_clf():
    print('RF classifier...')
    clf = RandomForestClassifier(random_state=0)
    param_dist = {
        'max_depth': randint(1, 4),
        'max_features': randint(1, 11),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'n_estimators': randint(30, 150)}
    return clf, param_dist


def lr_clf():
    print('Logistic regression...')
    clf = LogisticRegression(max_iter=100, random_state=0, tol=1e-4)
    param_dist = {'C': uniform(0.1, 20), 'dual': [True, False]}
    return clf, param_dist


def gbdt_clf():
    print('GBDT classifier...')
    clf = GradientBoostingClassifier(random_state=0, verbose=True)
    param_dist = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.01, 0.03, 0.1, 0.3],
        'n_estimators': randint(20, 200),
        'max_depth': randint(1, 6)}

    return clf, param_dist


def main():
    data = Data()
    train_X, train_y = data.get_train_set()
    test_X, test_y = data.get_test_set()
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)

    # hyper params search
    # clf, param_dist = lr_clf()
    # search_iters = 30
    # random_search = RandomizedSearchCV(
    #     clf, param_distributions=param_dist, n_iter=search_iters,
    #     verbose=True, cv=5)
    # start_time = time()
    # random_search.fit(train_X, train_y)
    # print('random search took %.2f seconds for %d candidates settings.' %
    #       (time() - start_time, search_iters))
    # cv_report(random_search.cv_results_)

    # ----------
    # RF
    # ----------
    # clf = RandomForestClassifier(
    #     max_depth=4, criterion='gini', random_state=0, n_estimators=45)
    # train_X, val_X, train_y, val_y = \
    #     train_test_split(train_X, train_y, train_size=0.8, random_state=0)
    # clf.fit(train_X, train_y)
    # fi = clf.feature_importances_
    # fn = data.feature_names
    # sorted_idx = np.argsort(fi)[::-1]
    # for i in sorted_idx:
    #     print('%s: %.4f' % (fn[i], fi[i]))
    # print(clf.score(train_X, train_y))
    # print(clf.score(val_X, val_y))

    # ----------
    # LR
    # ----------
    clf = LogisticRegression(dual=False, C=1.0, random_state=0)
    # train_X, val_X, train_y, val_y = \
    #     train_test_split(train_X, train_y, train_size=0.8, random_state=0)
    clf.fit(train_X, train_y)
    print(clf.score(train_X, train_y))
    # print(clf.score(val_X, val_y))
    print(clf.score(test_X, test_y))

    # ----------
    # GBDT
    # ----------
    # clf = GradientBoostingClassifier(
    #     max_depth=3, random_state=0, n_estimators=70, min_samples_split=4, min_samples_leaf=1)
    # train_X, val_X, train_y, val_y = \
    #     train_test_split(train_X, train_y, train_size=0.8, random_state=0)
    # clf.fit(train_X, train_y)
    # fi = clf.feature_importances_
    # fn = data.feature_names
    # sorted_idx = np.argsort(fi)[::-1]
    # for i in sorted_idx:
    #     print('%s: %.4f' % (fn[i], fi[i]))
    # print(clf.score(train_X, train_y))
    # print(clf.score(val_X, val_y))

    # ----------
    # SVM
    # ----------
    # clf = svm.SVC(verbose=True, kernel='linear', C=1.0, random_state=0,
    #               tol=1e-4)
    # train_X, val_X, train_y, val_y = \
    #     train_test_split(train_X, train_y, train_size=0.8, random_state=0)
    # clf.fit(train_X, train_y)
    # print(clf.score(train_X, train_y))
    # print(clf.score(val_X, val_y))

    # ----------
    # Neural Net model
    # ----------
    # nn = NeuralNet(data.num_features, data.num_outputs)
    # nn.fit(train_X, train_y)
    # print(nn.score(test_X, test_y))


if __name__ == '__main__':
    main()
