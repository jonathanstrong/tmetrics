from nose.tools import *
import tmetrics
import theano, theano.tensor as T
import numpy as np
import pandas as pd
import lasagne
import sklearn.metrics
from scipy.spatial.distance import hamming, jaccard, kulsinski
import nose.tools
from theano.tensor.tests.test_basic import makeTester


def setup():
    pass

def teardown():
    pass

def test_vector_clf_curve():
    yt = T.ivector('yt')
    yp = T.fvector('yp')
    tps = tmetrics.classification._vector_clf_curve(yt, yp)
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
    true = np.random.binomial(n=1, p=.5, size=1000).astype('int32')
    #true = np.array([0, 0, 1, 1]).astype('int32')
    predicted = np.random.random(size=1000).astype('float32')
    #predicted = np.array([0.1, 0.4, 0.35, 0.8]).astype('float32')
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
        fps, tps, thresholds = tmetrics.classification._binary_clf_curve(yt, yp)
        fpr, tpr, _thresh = tmetrics.classification.roc_curve(yt, yp)
        f = theano.function([yt, yp], [fps, tps, thresholds, fpr, tpr, _thresh, roc_auc_score_expr])
        result = f(true, predicted)
        print '** tmetrics **'
        print 'fps'
        print result[0]
        print 'tps'
        print result[1]
        print 'thresholds'
        print result[2]
        print 'fpr'
        print result[3]
        print 'tpr'
        print result[4]
        print '_thresh'
        print result[5]
        print 'roc score'
        print results[6]

        print '** refscore **'
        curve = sklearn.metrics.ranking._binary_clf_curve(true, predicted)
        print 'fpr'
        print curve[0]
        print 'tpr'
        print curve[1]
        print 'thresholds'
        print curve[2]
        trapz = np.trapz(curve[1], curve[0])
        print 'trapz'
        print trapz
        print 'auc'
        print sklearn.metrics.ranking.auc(curve[0], curve[1])
        print 'roc_curve'
        print sklearn.metrics.roc_curve(true, predicted)
        raise

def test_that_we_can_work_around_lexsort():
    x = np.asarray([0., 1., 1., 2.])
    y = np.asarray([1., 1., 2., 2.])
    order = np.lexsort((y, x))
    x1, y1 = x[order], y[order]
    sort_one = y.argsort()
    x2, y2 = x[sort_one], y[sort_one]
    sort_two = x2.argsort()
    x3, y3 = x2[sort_two], y2[sort_two]
    print 'x: {}; y: {}'.format(x, y)
    print 'x1: {}; y1: {}'.format(x1, y1)
    print 'x2: {}; y2: {}'.format(x2, y2)
    print 'x3: {}; y3: {}'.format(x3, y3)
    assert np.allclose(x1, x3)
    assert np.allclose(y1, y3)

""" 
def test_roc_curve_nd():
    true = np.random.binomial(n=1, p=.5, size=(2, 5)).astype('int32')
    predicted = np.random.random((2, 5)).astype('float32')
    yt = T.imatrix('yt')
    yp = T.fmatrix('yp')
    clf_curve = tmetrics.classification._binary_clf_curve_nd(yt, yp)
    f = theano.function([yt, yp], clf_curve)
    print 'true'
    print true
    print 'predicted'
    print predicted
    print 'tmetrics.classification._binary_clf_curve_nd(true, predicted)'
    print f(true, predicted)
    #print 'tmetrics.classification.roc_curve(true, predicted)'
    #print f(true, predicted)
    #assert False
"""


def test_matrix_roc_auc_scores():
    true = np.random.binomial(n=1, p=.5, size=(20, 100)).astype('int32')
    predicted = np.random.random((20, 100)).astype('float32')
    yt, yp = T.imatrix('yt'), T.fmatrix('yp')
    refscore = tmetrics.classification.last_axis_roc_auc_scores(true, predicted)
    roc_auc_scores = tmetrics.classification.roc_auc_scores(yt, yp)
    f = theano.function([yt, yp], roc_auc_scores)
    score = f(true, predicted)
    print 'refscore'
    print refscore
    print 'score'
    print score
    assert np.allclose(refscore, score)

def test_tensor3_roc_auc_scores():
    true = np.random.binomial(n=1, p=.5, size=(20, 30, 40)).astype('int32')
    predicted = np.random.random((20, 30, 40)).astype('float32')
    yt, yp = T.itensor3('yt'), T.ftensor3('yp')
    refscore = tmetrics.classification.last_axis_roc_auc_scores(true, predicted)
    roc_auc_scores = tmetrics.classification.roc_auc_scores(yt, yp)
    f = theano.function([yt, yp], roc_auc_scores)
    score = f(true, predicted)
    print 'refscore'
    print refscore
    print 'score'
    print score
    assert np.allclose(refscore, score, equal_nan=True)

def test_tensor4_roc_auc_scores():
    true = np.random.binomial(n=1, p=.5, size=(20, 30, 40, 50)).astype('int32')
    predicted = np.random.random((20, 30, 40, 50)).astype('float32')
    yt, yp = T.itensor4('yt'), T.ftensor4('yp')
    refscore = tmetrics.classification.last_axis_roc_auc_scores(true, predicted)
    roc_auc_scores = tmetrics.classification.roc_auc_scores(yt, yp)
    f = theano.function([yt, yp], roc_auc_scores)
    score = f(true, predicted)
    print 'refscore'
    print refscore
    print 'score'
    print score
    assert np.allclose(refscore, score, equal_nan=True)

@nose.tools.raises(TypeError)
def test_roc_curves_exception_if_numpy_object_passed():
    y = np.array([0, 0, 1, 1]).astype('int32')
    scores = np.array([0.1, 0.4, 0.35, 0.8]).astype('float32')
    fpr, tpr, thresh = tmetrics.classification.roc_curves(y, scores)

@nose.tools.raises(ValueError)
def test_roc_curves_dimension_checker():
    y = T.imatrix('y')
    p = T.ftensor3('p')
    fpr, tpr, _ = tmetrics.classification.roc_curves(y, p)

@nose.tools.raises(TypeError)
def test_roc_auc_scores_exception_if_numpy_object_passed():
    y = np.array([0, 0, 1, 1]).astype('int32')
    scores = np.array([0.1, 0.4, 0.35, 0.8]).astype('float32')
    fpr, tpr, thresh = tmetrics.classification.roc_auc_scores(y, scores)

@nose.tools.raises(ValueError)
def test_roc_auc_scores_dimension_checker():
    y = T.imatrix('y')
    p = T.ftensor3('p')
    fpr, tpr, _ = tmetrics.classification.roc_auc_scores(y, p)


def test_1D_roc_auc_scores():
    yt = T.ivector('yt')
    yp = T.fvector('yp')
    y = np.array([0, 0, 1, 1]).astype('int32')
    scores = np.array([0.1, 0.4, 0.35, 0.8]).astype('float32')
    ref_fpr, ref_tpr, ref_thresh = sklearn.metrics.roc_curve(y, scores)
    roc_auc_scores = tmetrics.classification.roc_auc_scores(yt, yp)
    fpr, tpr, thresh = tmetrics.classification.roc_curves(yt, yp)
    f = theano.function([yt, yp], [fpr, tpr, thresh, roc_auc_scores])
    score_fpr, score_tpr, score_thresh, score_auc = f(y ,scores)
    assert np.allclose(ref_fpr, score_fpr)
    assert np.allclose(ref_tpr, score_tpr)
    assert np.allclose(ref_thresh, score_thresh)
    assert np.allclose(sklearn.metrics.roc_auc_score(y, scores), score_auc)

def test_roc_scores_slogan():
    "returns roc curves calculated axis[-1]-wise"
    yt = T.itensor4('yt')
    yp = T.ftensor4('yp')
    true = np.random.binomial(n=1, p=.5, size=(2, 3, 4, 5)).astype('int32')
    predicted = np.random.random((2, 3, 4, 5)).astype('float32')
    fpr_e, tpr_e, _e = tmetrics.classification.roc_curves(yt, yp)
    scores_expr = tmetrics.classification.roc_auc_scores(yt, yp)
    f = theano.function([yt, yp], [fpr_e, tpr_e, _e, scores_expr])
    fpr, tpr, _, scores = f(true, predicted)
    print true.shape, predicted.shape, fpr.shape, scores.shape
    print true
    print predicted
    print fpr
    print tpr
    print _
    print scores
    assert true.shape == predicted.shape == fpr.shape == tpr.shape == _.shape
    assert true.shape[:-1] == scores.shape


def test_precisison_recall_curves_vector(n_iter=1):
    yt = T.ivector('yt')
    yp = T.fvector('yp')
    p_expr, r_expr, thresh_expr = tmetrics.classification.precision_recall_curves(yt, yp)
    f = theano.function([yt, yp], [p_expr, r_expr, thresh_expr])
    for iterator in xrange(n_iter):
        y = np.random.binomial(n=1, p=.5, size=20).astype('int32')
        scores = np.random.random(20).astype('float32')
        ref_precision, ref_recall, ref_thresh = sklearn.metrics.precision_recall_curve(y, scores)
        precision, recall, thresh = f(y ,scores)
        #assert np.allclose(ref_precision, precision)
        #assert np.allclose(ref_recall, recall)
        #assert np.allclose(ref_thresh, thresh)
        try:
            assert np.allclose(sklearn.metrics.auc(ref_recall, ref_precision), sklearn.metrics.auc(recall, precision))
        except:
            print 'n_iter: {}'.format(n_iter)
            print 'y'
            print y
            print 'scores'
            print scores
            print 'ref precision'
            print ref_precision
            print ref_precision.shape
            #print np.r_[precision[1:], 1] 
            #print np.allclose(ref_precision, np.r_[precision[1:], 1] )
            print sklearn.metrics.auc(ref_recall, ref_precision)
            print sklearn.metrics.auc(recall, precision)
            print
            print 'ref recall'
            print ref_recall
            print ref_recall.shape
            print
            print 'ref thresh'
            print ref_thresh
            print ref_thresh.shape
            print
            print 'score precision'
            print precision
            print precision.shape
            print
            print 'score recall'
            print recall
            print recall.shape
            print 
            print 'score threshold'
            print thresh
            print thresh.shape
            raise


def test_last_axis_precision_recall_curve():
    dims = [2, 3, 4, 5]
    for ndim in [1, 2, 3, 4]:
        if ndim == 1: 
            d = 10
        else:
            d = dims[:ndim]
        a = np.random.binomial(n=1, p=.5, size=d)
        p = np.random.random(d)
        prec, recall, _ = tmetrics.last_axis_precision_recall_curve(a, p)
        print 'd: {}'.format(d)
        print prec
        print recall
        print _

def test_precision_recall_curves_all_dims(n_iter=1):
    dims = [(1000,), (500, 100), (50, 300, 400), (20, 50, 100, 30)]
    int_types = [T.ivector, T.imatrix, T.itensor3, T.itensor4]
    float_types = [T.fvector, T.fmatrix, T.ftensor3, T.ftensor4]
    for d, int_type, float_type in zip(dims[1:], int_types[1:], float_types[1:]):
        yt = int_type('yt')
        yp = float_type('yp')
        a = float_type('a')
        b = float_type('b')
        gpu_auc = tmetrics.auc(a, b)
        get_auc = theano.function([a, b], gpu_auc)
        p_expr, r_expr, t_expr = tmetrics.precision_recall_curves(yt, yp)
        pr_auc = tmetrics.auc(r_expr, p_expr) 
        f = theano.function([yt, yp], [p_expr, r_expr, t_expr, pr_auc])
        for epoch in xrange(n_iter):
            true = np.random.binomial(n=1, p=.5, size=d).astype('int32')
            predicted = np.random.random((d)).astype('float32')
            precision, recall, thresh, avg = f(true, predicted)
            refp, refr, reft = tmetrics.last_axis_precision_recall_curve(true, predicted)
            try:
                assert np.allclose(precision, refp, equal_nan=True)
                assert np.allclose(recall, refr, equal_nan=True)
                assert np.allclose(thresh, reft, equal_nan=True)
                assert np.allclose(avg, get_auc(refr.astype('float32'), refp.astype('float32')), equal_nan=True)
            except:
                print true
                print predicted
                print 'precision'
                print precision
                print 'ref precision'
                print refp
                print 'recall'
                print recall
                print recall.shape
                print 'ref recall'
                print refr
                print refr.shape
                print thresh
                print reft
                print avg
                print get_auc(refr.astype('float32'), refp.astype('float32'))
                raise




