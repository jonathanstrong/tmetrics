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

    #also test the function is numpy/pandas compatible
    assert (refscore - tol) < tmetrics.brier_score_loss(yt, yp) < (refscore + tol)


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

def test_trapz():
    x = T.ivector('x')
    y = T.ivector('y')
    z = tmetrics.classification.trapz(y, x)
    refscore = np.trapz([1,2,3], x=[4,6,8])
    f = theano.function([y, x], z, allow_input_downcast=True)
    score = f([1,2,3], [4,6,8])
    assert np.allclose(refscore, score)
    refscore = np.trapz([1,2,3])
    z = tmetrics.classification.trapz(y)
    f = theano.function([y], z, allow_input_downcast=True)
    score = f([1,2,3])
    assert np.allclose(refscore, score)
    refscore = np.trapz([1,2,3], dx=2)
    z = tmetrics.classification.trapz(y, dx=2)
    f = theano.function([y], z, allow_input_downcast=True)
    score = f([1,2,3])
    assert np.allclose(score, refscore)
    n = T.fmatrix('n')
    m = T.fmatrix('m')
    a = np.arange(6).reshape(2,3)
    refscore = np.trapz(a, axis=0)
    z = tmetrics.classification.trapz(n, axis=0)
    f = theano.function([n], z, allow_input_downcast=True)
    score = f(a)
    assert np.allclose(score, refscore)
    refscore = np.trapz(a, axis=1)
    z = tmetrics.classification.trapz(n, axis=1)
    f = theano.function([n], z, allow_input_downcast=True)
    score = f(a)
    assert np.allclose(score, refscore)

    
def test_roc_auc_score():
    #true = np.random.binomial(n=1, p=.5, size=1000).astype('int32')
    true = np.array([0, 0, 1, 1]).astype('int32')
    #predicted = np.random.random(size=1000).astype('float32')
    predicted = np.array([0.1, 0.4, 0.35, 0.8]).astype('float32')
    yt = T.ivector('y_true')
    yp = T.fvector('y_predicted')
    roc_auc_score_expr = tmetrics.classification.roc_auc_score(yt, yp)
    refscore = sklearn.metrics.roc_auc_score(true, predicted)
    print 'refscore'
    print refscore
    f = theano.function([yt, yp], roc_auc_score_expr)
    score = f(true, predicted)
    print 'score'
    print score
    try:
        assert np.allclose(refscore, score)
    except AssertionError:
        fpr, tpr, thresholds = tmetrics.classification._binary_clf_curve(yt, yp)
        trapz = tmetrics.classification.trapz(fpr, tpr)
        f = theano.function([yt, yp], [fpr, tpr, thresholds, trapz])
        result = f(true, predicted)
        print '** tmetrics **'
        print 'fpr'
        print result[0]
        print 'tpr'
        print result[1]
        print 'thresholds'
        print result[2]
        print 'trapz'
        print result[3]

        print '** refscore **'
        curve = sklearn.metrics.ranking._binary_clf_curve(true, predicted)
        print 'fpr'
        print curve[0]
        print 'tpr'
        print curve[1]
        print 'thresholds'
        print curve[2]
        trapz = np.trapz(curve[0], curve[1])
        print 'trapz'
        print trapz
        raise



