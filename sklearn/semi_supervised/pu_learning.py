# coding=utf8
"""
PU learning, or Positive Unlabeled learning, refers to the semisupervised
classification problem where some positive labels are known and the negative
labels contain a mixture of positive and negative labels. The setting commonly
arises when domain experts record only the data that is of interest to them.

For more information see the references below.

Model Features
--------------
Feat1:
    Descr

Feat2:
    Descr

Examples
--------
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import POSOnly
>>> from sklearn.svm import SVC
>>> clf = SVC()
>>> pu_clf = POSOnly(clf)
>>> x, y = datasets.make_classification()
>>> #TODO
>>> # finish this
... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
PUClassifier(...)

Notes
-----
References:
[1]  Charles Elkan, Keith Noto. Learning Classifiers from Only Positive and
Unlabeled Data. ACM 2008
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.9201
"""

# Authors: Jacob Mick <jam7w2@mail.missouri.edu>
# Licence: BSD
from .base import BaseEstimator
import numpy as np


class POSOnly(BaseEstimator):
    """ POSOnly Classifier

    Parameters
    ----------
    estimator
        descr

    hold_out_ratio
        descr

    precomputed_kernel
        descr

    random_state
        descr

    Adapts any probabilistic binary classifier to Positive Unlabled Learning
    using the PosOnly method proposed by Elkan and Noto.

    Notes
    -----
    References:
    [1]  Charles Elkan, Keith Noto. Learning Classifiers from Only Positive and
    Unlabeled Data. ACM 2008
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.9201

    """

    def __init__(self, estimator, hold_out_ratio=0.1, precomputed_kernel=False,
                 random_state=None):
        """
        estimator -- An estimator of p(s=1|x) that must implement:
                     * predict_proba(X): Takes X, which can be a list of feature vectors or a precomputed
                                         kernel matrix and outputs p(s=1|x) for each example in X
                     * fit(X,y): Takes X, which can be a list of feature vectors or a precomputed
                                 kernel matrix and takes y, which are the labels associated to the
                                 examples in X
        hold_out_ratio -- The ratio of training examples that must be held out of the training set of examples
                          to estimate p(s=1|y=1) after training the estimator
        precomputed_kernel -- Specifies if the X matrix for predict_proba and fit is a precomputed kernel matrix
        """
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio

        if not hasattr(estimator, 'fit') or not hasattr(estimator, 'predict_proba'):
            raise TypeError("The underlying estimator for POSOnly doesn't"
                            "implement methods fit and predict."
                            "'%s'" % (estimator,))

        if precomputed_kernel:
            self.fit = self.__fit_precomputed_kernel
        else:
            self.fit = self.__fit_no_precomputed_kernel

        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator:' + str(self.estimator) + '\n' + 'p(s=1|y=1,x) ~= ' + str(self.c) + '\n' + \
            'Fitted: ' + str(self.estimator_fitted)

    def __fit_precomputed_kernel(self, X, y):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1) using a subset of the training examples

        X -- Precomputed kernel matrix
        y -- Labels associated to each example in X (Positive label: 1.0, Negative label: -1.0)
        """
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)

        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]

        #Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        # TODO
        # use np.setdiff1d() here
        keep = list(set(np.arange(len(y))) - set(hold_out))
        X_test_hold_out = X_test_hold_out[:, keep]

        #New training kernel matrix
        X = X[:, keep]
        X = X[keep]

        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_test_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:, 1]
        except:
            pass

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True

    def __fit_no_precomputed_kernel(self, X, y):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1,x)

        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        # TODO
        # self.classes_, y = unique(y, return_inverse=True)
        # Don't ask for a particular formulation of y.
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)

        #TODO
        # what kind of error should this raise
        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out, 0)
        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:, 1]
        except:
            pass

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True

    def predict_proba(self, X):
        """
        Predicts p(y=1|x) using the estimator and the value of p(s=1|y=1) estimated in fit(...)

        X -- List of feature vectors or a precomputed kernel matrix
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        probabilistic_predictions = self.estimator.predict_proba(X)

        try:
            probabilistic_predictions = probabilistic_predictions[:, 1]
        except:
            pass

        return probabilistic_predictions / self.c

    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else -1. for p in self.predict_proba(X)])
