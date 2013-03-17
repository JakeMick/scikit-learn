# coding=utf8
"""
PU learning, or Positive Unlabeled learning, refers to the semisupervised
classification problem where some positive labels are known and the negative
labels contain a mixture of positive and negative labels. The setting commonly
arises when domain experts record only the data that is of interest to them.

In order for PU learning to be successful the flipping of positive labels to
the unlabeled class must be independent of the training features. The problem
then reduces to calibration of probability estimates. POSOnly finds this
calibration by scaling the ratio of observed positives/unlabeled to the
ratio of predicted postives/negatives on held-out data.

For more information see the references below.

Model Features
--------------
Feat1:
    Descr

Feat2:
    Descr

Examples
--------
#TODO
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

# Authors: Alexandre Drouin <alexandre.drouin.8@ulaval.ca>,
#          Jacob Mick <jam7w2@mail.missouri.edu>
# Licence: BSD

from ..base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
import numpy as np


class POSOnly(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """ POSOnly Classifier

    Parameters
    ----------
    estimator : estimator object
        A classifier that implements fit and predict_proba.

    held_out_ratio : float, default:0.1
        Fraction of samples to use when computing the probability that a positive
        example is labeled.
        1 - held_out_ratio is the number of observations to use when fitting
        the underlying classifier.

    estimator_input_type : str, default:'feature_vector'
        Specifies the type of the data used when fitting and predicting with
        the underlying classifier. It must be 'kernel_matrix' or 'feature_vector'.

    random_state : int or RandomState
        Random seed used for shuffling the data before splitting it.

    Adapts any probabilistic binary classifier to Positive Unlabled Learning
    using the PosOnly method proposed by Elkan and Noto.

    Notes
    -----
    The class with the larger numerical value is assumed to be the postive
    class. The class with the smaller numerical value is assumed to be the
    negative class.

    References
    ----------
    [1]  Charles Elkan, Keith Noto. Learning Classifiers from Only Positive and
    Unlabeled Data. ACM 2008
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.9201

    """
    # TODO
    # use util funcs to check assumptions
    # e.g. 2darray, sparse arrays...
    def __init__(self, estimator, held_out_ratio=0.1, estimator_input_type='feature_vector',
                 random_state=None):
        self.estimator = estimator
        self.held_out_ratio = held_out_ratio
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = None

        if not hasattr(estimator, 'fit') or not hasattr(estimator, 'predict_proba'):
            raise TypeError("The underlying estimator for POSOnly doesn't"
                            "implement methods fit and predict_proba."
                            "'%s'" % (estimator,))

        if estimator_input_type == 'kernel_matrix':
            self.fit = self._fit_kernel_matrix
        else:
            self.fit = self._fit_feature_vectors

        self.estimator_fitted = False

    def _fit_kernel_matrix(self, X, y):
        positives = np.where(y == 1)[0]
        held_out_size = np.ceil(positives.shape[0] * self.held_out_ratio)

        if self.random_state is not None:
            np.random.seed(seed=self.random_state)
        np.random.shuffle(positives)

        held_out = positives[:held_out_size]

        #Hold out test kernel matrix
        X_test_held_out = X[held_out]

        # XXX
        # Can we allocated np.arange(y.shape[0]) at a higher level?
        keep = np.setdiff1d(np.arange(y.shape[0]), held_out)
        X_test_held_out = X_test_held_out[:, keep]

        #New training kernel matrix
        X = X[:, keep]
        X = X[keep]

        y = np.delete(y, held_out)

        self.estimator.fit(X, y)

        held_out_predictions = self.estimator.predict_proba(X_test_held_out)

        #TODO
        # I understand why this is here.
        # Some probabilistic classifiers in sklearn have that awkward 2d
        # [p, 1-p] prediction. There's got to be a better way to handle this
        # than a try-except pass.
        try:
            held_out_predictions = held_out_predictions[:, 1]
        except:
            pass

        c = np.mean(held_out_predictions)
        self.c = c

        self.estimator_fitted = True

    def _fit_feature_vectors(self, X, y):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1,x)

        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        # XXX

        # Assuming that the class with the larger value is the positive class.
        # Should we ask for this explicitly in the docs?
        self.classes_, y = np.unique(y, return_inverse=True)
        positives = np.where(y == 1.)[0]
        held_out_size = np.ceil(positives.shape[0] * self.held_out_ratio)

        if self.random_state is not None:
            np.random.seed(seed=self.random_state)
        np.random.shuffle(positives)

        held_out = positives[:held_out_size]
        X_held_out = X[held_out]
        X = np.delete(X, held_out, 0)
        y = np.delete(y, held_out)

        self.estimator.fit(X, y)

        held_out_predictions = self.estimator.predict_proba(X_held_out)

        #TODO
        # I understand why this is here.
        # Some probabilistic classifiers in sklearn have that awkward 2d
        # [p, 1-p] prediction. There's got to be a better way to handle this
        # than a try-except pass.
        try:
            held_out_predictions = held_out_predictions[:, 1]
        except:
            pass

        c = np.mean(held_out_predictions)
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

        #TODO
        # I understand why this is here.
        # Some probabilistic classifiers in sklearn have that awkward 2d
        # [p, 1-p] prediction. There's got to be a better way to handle this
        # than a try-except pass.

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
            raise Exception("The estimator must be fitted before calling predict(...).")

        #TODO
        # Need to map the prediction to the origin self.classes_
        return (self.predict_proba(X) > self.threshold).astype('int')