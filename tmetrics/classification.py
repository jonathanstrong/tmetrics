import theano, theano.tensor as T
import numpy as np
import pandas as pd
import lasagne


"""
note: we are following the sklearn api for metrics/loss functions,
where the first arg for a function is y true, and second value is
y predicted. this is the opposite of the theano functions, so just
keep in mind.

"""

#copy existing code and place in tmetrics namespace
multiclass_hinge_loss = lambda yt, yp: lasagne.objectives.multiclass_hinge_loss(yp, yt)
squared_error = lambda yt, yp: lasagne.objectives.squared_error(yp, yt)
binary_accuracy = lambda yt, yp: lasagne.objectives.binary_accuracy(yp, yt)
categorical_accuracy = lambda yt, yp: lasagne.objectives.categorical_accuracy(yp, yt)

def binary_crossentropy(y_true, y_predicted):
    """
    wrapper of theano.tensor.nnet.binary_crossentropy
    args reversed to match tmetrics api
    """
    return T.nnet.binary_crossentropy(y_predicted, y_true)

def categorical_crossentropy(y_true, y_predicted):
    """
    wrapper of theano.tensor.nnet.categorical_crossentropy
    args reversed to match tmetrics api
    """
    return T.nnet.binary_crossentropy(y_predicted, y_true)

def binary_hinge_loss(y_true, y_predicted, binary=True, delta=1):
    """
    wrapper of lasagne.objectives.binary_hinge_loss
    args reversed to match tmetrics api
    """
    return lasagne.objectives.binary_hinge_loss(y_predicted, y_true, binary, delta)




def brier_score_loss(y_true, y_predicted, sample_weight=None):
    """
    port of sklearn.metrics.brier_score_loss
    works for 2D binary data as well, e.g.

    y_true:          [[0, 1, 0],
                     [1, 0, 0]]

    y_predicted:    [[.1, .9, .3],
                     [.4, .7, .2]]

    y_true: tensor, y true (binary)
    y_predicted: tensor, y predicted (float between 0 and 1)  
    sample_weight: tensor or None (standard mean)

    assumptions: 
     -binary ground truth values ({0, 1}); no pos_label
        training wheels like sklearn or figuring out how to 
        run this on text labels. 
     -probabilities are floats between 0-1
     -sample_weight broadcasts to ((y_true - y_predicted) ** 2)

    """
    scores = ((y_true - y_predicted) ** 2)
    if sample_weight is not None: 
        scores = scores * sample_weight
    return scores.mean()

def _binary_clf_curve(y_true, y_predicted):
    """
    sklearn.metrics._binary_clf_curve port

    y_true: tensor (ivector): y true
    y_predicted: tensor (fvector): y predicted

    returns: fps, tps, threshold_values
    fps: tensor (ivector): false positivies
    tps: tensor (ivector): true positives
    threshold_values: tensor (fvector): value of y predicted at each threshold 
        along the curve

    restrictions: 
        -not numpy compatible
        -only works with two vectors (not matrix or tensor)


    """
    desc_score_indices = y_predicted.argsort()[::-1]
    sorted_y_predicted = y_predicted[desc_score_indices]
    sorted_y_true = y_true[desc_score_indices]

    distinct_value_indices = (1-T.isclose(T.extra_ops.diff(sorted_y_predicted), 0)).nonzero()[0]
    curve_cap = T.extra_ops.repeat(sorted_y_predicted.size - 1, 1)
    threshold_indices = T.concatenate([distinct_value_indices, curve_cap])

    tps = T.extra_ops.cumsum(sorted_y_true[threshold_indices])
    fps = 1 + threshold_indices - tps
    threshold_values = sorted_y_predicted[threshold_indices]

    return fps, tps, threshold_values

def hamming_loss(y_true, y_predicted):
    """
    note - works on n-dim arrays, means across the final axis

    note - we round predicted because float probabilities would not work
    """
    return T.neq(y_true, T.round(y_predicted)).mean(axis=-1)

def jaccard_similarity(y_true, y_predicted):
    """
    y_true: tensor ({1, 0})
    y_predicted: tensor ({1, 0})
    note - we round predicted because float probabilities would not work
    """
    y_predicted = T.round(y_predicted).astype(theano.config.floatX)
    either_nonzero = T.or_(T.neq(y_true, 0), T.neq(y_predicted, 0))
    return T.and_(T.neq(y_true, y_predicted), either_nonzero).sum(axis=-1, dtype=theano.config.floatX) / either_nonzero.sum(axis=-1, dtype=theano.config.floatX)
        
        
def _nbool_correspond_all(u, v):
    """
    port of scipy.spatial.distance._nbool_correspond_all
    with dtype assumed to be integer/float (no bool in theano)

    sums are on last axis

    """
    not_u = 1.0 - u
    not_v = 1.0 - v
    nff = (not_u * not_v).sum(axis=-1, dtype=theano.config.floatX)
    nft = (not_u * v).sum(axis=-1, dtype=theano.config.floatX)
    ntf = (u * not_v).sum(axis=-1, dtype=theano.config.floatX)
    ntt = (u * v).sum(axis=-1, dtype=theano.config.floatX)
    return (nff, nft, ntf, ntt)

def kulsinski_similarity(y_true, y_predicted):
    y_predicted = T.round(y_predicted)
    nff, nft, ntf, ntt = _nbool_correspond_all(y_true, y_predicted)
    n = y_true.shape[0].astype('float32')
    return (ntf + nft - ntt + n) / (ntf + nft + n)
        
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    port from numpy.trapz ... pretty much exact function.

    ...

    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.
    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        If `x` is None, then spacing between all `y` elements is `dx`.
    dx : scalar, optional
        If `x` is None, spacing given by `dx` is assumed. Default is 1.
    axis : int, optional
        Specify the axis.
    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule.
    See Also
    --------
    sum, cumsum
    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.
    References
    ----------
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule
    .. [2] Illustration image:
           http://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png
    Examples
    --------
    >>> np.trapz([1,2,3])
    4.0
    >>> np.trapz([1,2,3], x=[4,6,8])
    8.0
    >>> np.trapz([1,2,3], dx=2)
    8.0
    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> np.trapz(a, axis=1)
    array([ 2.,  8.])
    """
    if x is None:
        d = dx
    else:
        if x.ndim == 1:
            d = T.extra_ops.diff(x)
            # reshape to correct shape
            shape = T.ones(y.ndim, dtype='int32')
            shape = T.set_subtensor(shape[axis], d.shape[0])
            d = d.reshape(shape)
        else:
            d = T.extra_ops.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    return (d * (y[slice1] + y[slice2]) / 2.0).sum(axis)

def auc(x, y):
    return trapz(y, x)

def roc_auc_score(y_true, y_predicted):
    fpr, tpr, thresholds = _binary_clf_curve(y_true, y_predicted)
    return auc(fpr, tpr)

