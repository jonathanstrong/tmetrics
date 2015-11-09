from nose.tools import *
import tmetrics
import theano, theano.tensor as T
import numpy as np
import pandas as pd
import lasagne
import sklearn.metrics
from scipy.spatial.distance import hamming, jaccard, kulsinski


def setup():
    pass

def teardown():
    pass

def test_binary_clf_curve():
    yt = T.ivector('yt')
    yp = T.fvector('yp')
    tps = tmetrics.classification._binary_clf_curve(yt, yp)
    f = theano.function([yt, yp], tps, allow_input_downcast=True)
    true, predicted = np.random.binomial(n=1, p=.5, size=10), np.random.random(10)
    fps, tps, _ = f(true, predicted)
    s_fps, s_tps, s_ = sklearn.metrics.ranking._binary_clf_curve(true, predicted)
    np.set_printoptions(suppress=True)
    print 'true'
    print true
    print 'predicted'
    print predicted
    print 'fps'
    print fps
    print 'sklearn fps'
    print s_fps
    print 'tps'
    print tps
    print 'sklearn tps'
    print s_tps
    print 'threshold values'
    print _
    print 'sklearn threshold values'
    print s_
    assert np.allclose(fps, s_fps)
    assert np.allclose(tps, s_tps)
    assert np.allclose(_, s_)


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


def test_hammming_loss():
    true = np.random.binomial(n=1, p=.5, size=10)
    predicted = np.round(np.random.random(10))
    refscore = hamming(true, predicted)
    yt = T.fvector('yt')
    yp = T.fvector('yp')
    f = theano.function([yt, yp], tmetrics.classification.hamming_loss(yt, yp), allow_input_downcast=True)
    score = f(true, predicted)
    print 'true'
    print true
    print 'predicted'
    print predicted
    print 'refscore {}'.format(refscore)
    print 'score {}'.format(score)
    assert np.allclose(refscore, score)

def test_jaccard_similarity():
    true = np.random.binomial(n=1, p=.5, size=10)
    predicted = np.round(np.random.random(10))
    refscore = jaccard(true, predicted)
    yt = T.fvector('yt')
    yp = T.fvector('yp')
    f = theano.function([yt, yp], tmetrics.classification.jaccard_similarity(yt, yp), allow_input_downcast=True)
    score = f(true, predicted)
    print 'true'
    print true
    print 'predicted'
    print predicted
    print 'refscore {}'.format(refscore)
    print 'score {}'.format(score)
    assert np.allclose(refscore, score)

def test_jaccard_similarity_2D():
    true = np.random.binomial(n=1, p=.5, size=10)
    predicted = np.round(np.random.random(10))
    refscore = np.asarray([jaccard(true, predicted)])
    double = lambda x: np.concatenate([x.reshape((1, len(x))), x.reshape((1, len(x)))])
    true, predicted, refscore = tuple(double(x) for x in [true, predicted, refscore])
    yt = T.fmatrix('yt')
    yp = T.fmatrix('yp')
    f = theano.function([yt, yp], tmetrics.classification.jaccard_similarity(yt, yp), allow_input_downcast=True)
    score = f(true, predicted)
    print 'true'
    print true
    print 'predicted'
    print predicted
    print 'refscore {}'.format(refscore)
    print 'score {}'.format(score)
    assert np.allclose(refscore, score)



def test_kulsinski_similarity():
    true = np.double(np.random.binomial(n=1, p=.5, size=10))
    predicted = np.double(np.round(np.random.random(10)))
    refscore = kulsinski(true, predicted)
    yt = T.fvector('yt')
    yp = T.fvector('yp')
    f = theano.function([yt, yp], tmetrics.classification.kulsinski_similarity(yt, yp), allow_input_downcast=True)
    score = f(true, predicted)
    print 'true'
    print true
    print 'predicted'
    print predicted
    print 'refscore {}'.format(refscore)
    print 'score {}'.format(score)
    assert np.allclose(refscore, score)





