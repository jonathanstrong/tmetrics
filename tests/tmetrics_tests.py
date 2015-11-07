from nose.tools import *
import tmetrics
import theano, theano.tensor as T
import numpy as np
import pandas as pd
import lasagne
import sklearn.metrics


def setup():
    pass

def teardown():
    pass

def testing_testing_123():
    assert 1 > 0

def test_brier_score_loss_from_scikit_learn_example():
    """
    from sklearn docs...
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    >>> brier_score_loss(y_true, y_prob)  
    0.037...

    """
    y_true = T.ivector('y_true')
    y_predicted = T.fvector('y_predicted')
    brier_score = tmetrics.brier_score_loss(y_true, y_predicted)
    f = theano.function([y_true, y_predicted], brier_score)
    yt = np.array([0, 1, 1, 0], 'int32')
    yp = np.array([.1, .9, .8, .3], theano.config.floatX)
    refscore = sklearn.metrics.brier_score_loss(yt, yp)
    tol = .01
    score = f(yt, yp)
    assert (refscore - tol) < score < (refscore + tol)


def test_brier_score_loss_2D():
    yt = np.array([0, 1, 1, 0], 'int32')
    yp = np.array([.1, .9, .8, .3], theano.config.floatX)
    refscore = sklearn.metrics.brier_score_loss(yt, yp)
    yt = np.concatenate([yt.reshape(1, 4), yt.reshape(1, 4)], axis=0)
    yp = np.concatenate([yp.reshape(1, 4), yp.reshape(1, 4)], axis=0)
    y_true = T.imatrix('y_true')
    y_predicted = T.fmatrix('y_predicted')
    brier_score = tmetrics.brier_score_loss(y_true, y_predicted)
    f = theano.function([y_true, y_predicted], brier_score)
    tol = .01
    score = f(yt, yp)
    assert (refscore - tol) < score < (refscore + tol)


