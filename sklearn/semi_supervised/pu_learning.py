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
        The kernel_matrix type must be used for underlying estimators with a
        kernel='precomputed' parameter.

    random_state : int or RandomState
        Random seed used for shuffling the data before splitting it.

    Adapts any probabilistic binary classifier to Positive Unlabled Learning
    using the PosOnly method proposed by Elkan and Noto.

    Notes
    -----
    The class with the larger numerical value is assumed to be the positive
    class. The class with the smaller numerical value is assumed to be the
    unlabeled class.

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

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X. The class with the largest numerical
            value is assumed to be the positive class. The class with the smallest
            numerical value is assumed to be the unlabeled class.

        Returns
        -------
        self : object
            Returns self.
        """
        return NotImplemented

    #TODO
    #This method was never properly tested. I implemented it quickly and ended
    #up never using it. Although, for kernel method users, this is a necessary
    #feature. We could write a test that computes a kernel matrix with the RBF
    #kernel and checks that the column/row holding out works properly.
    def _fit_kernel_matrix(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_samples]
            Precomputed kernel matrix

        y : array-like, shape = [n_samples]
            Target vector relative to X. The class with the largest numerical
            value is assumed to be the positive class. The class with the smallest
            numerical value is assumed to be the unlabeled class.

        Returns
        -------
        self : object
            Returns self.
        """
        # XXX
        # Assuming that the class with the larger value is the positive class.
        self.classes_, y = np.unique(y, return_inverse=True)

        if self.classes_.shape[0] != 2:
            raise ValueError('The target vector must contain exactly two classes.')

        positives = np.where(y == self.classes_[1])[0]
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

        y = y[keep]

        self.estimator.fit(X, y)

        held_out_predictions = self.estimator.predict_proba(X_test_held_out)

        if np.rank(held_out_predictions) > 1:
            held_out_predictions = held_out_predictions[:, 1]

        self.c = np.mean(held_out_predictions)
        self.estimator_fitted = True

    def _fit_feature_vectors(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_samples]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector relative to X. The class with the largest numerical
            value is assumed to be the positive class. The class with the smallest
            numerical value is assumed to be the unlabeled class.

        Returns
        -------
        self : object
            Returns self.
        """
        # XXX
        # Assuming that the class with the larger value is the positive class.
        self.classes_, y = np.unique(y, return_inverse=True)

        if self.classes_.shape[0] != 2:
            raise ValueError('The target vector must contain exactly two classes.')

        positives = np.where(y == self.classes_[1])[0]
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

        if np.rank(held_out_predictions) > 1:
            held_out_predictions = held_out_predictions[:, 1]

        self.c = np.mean(held_out_predictions)
        self.estimator_fitted = True

    def predict_proba(self, X):
        """Compute probability of having the positive class for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Feature vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array-like, shape = [n_samples]
            Returns the probability of the sample for the positive class.
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        probabilistic_predictions = self.estimator.predict_proba(X)

        if np.rank(probabilistic_predictions) > 1:
            probabilistic_predictions = probabilistic_predictions[:, 1]

        return probabilistic_predictions / self.c

    def predict(self, X, threshold=0.5):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Feature vectors, where n_samples is the number of samples
            and n_features is the number of features.
        threshold : float [0,1], default:0.5
            The threshold used to determine the class of an example based
            on its probability of having the positive class. All examples
            with probabilities above the threshold will be classified as
            having the positive class.

        Returns
        -------
        y : array, shape = [n_samples]
            Class labels for samples in X.
        """
        #TODO
        #Find less generic exception type
        if not self.estimator_fitted:
            raise Exception("The estimator must be fitted before calling predict(...).")

        return [self.classes_[x] for x in (self.predict_proba(X) > threshold).astype('int')]