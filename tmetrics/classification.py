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

def _vector_clf_curve(y_true, y_predicted):
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
    assert y_true.ndim == y_predicted.ndim == 1

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
      
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    reference implementation: numpy.trapz 
    ---------

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
    return abs(trapz(y, x))

#def roc_curve(y_true, y_predicted):
#    fps, tps, thresholds = _binary_clf_curve(y_true, y_predicted)
#    fpr = fps.astype('float32') / fps[-1]
#    tpr = tps.astype('float32') / tps[-1]
#    return fpr, tpr, thresholds
#
#def roc_auc_score(y_true, y_predicted):
#    fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
#    return auc(fpr, tpr)

"""
NUMPY ONLY FUNCTIONS
"""
def _last_axis_binary_clf_curve(y_true, y_predicted):
    """
    returns y_predicted.shape[-2] binary clf curves calculated axis[-1]-wise

    """
    assert y_true.shape == y_predicted.shape
    axis = -1
    sort_idx = list(np.ogrid[[slice(x) for x in y_predicted.shape]])
    sort_idx[axis] = y_predicted.argsort(axis=axis)
    reverse = [slice(None)] * y_predicted.ndim
    reverse[axis] = slice(None, None, -1)
    sorted_y_predicted = y_predicted[sort_idx][reverse]
    sorted_y_true = y_true[sort_idx][reverse]


    tps = sorted_y_true.cumsum(axis=axis)
    count = (np.ones(y_predicted.shape) * np.arange(y_predicted.shape[-1]))
    fps = 1 + count - tps
    threshold_values = sorted_y_predicted

    return fps, tps, threshold_values

def last_axis_roc_curve(y_true, y_predicted):
    fps, tps, thresholds = _last_axis_binary_clf_curve(y_true, y_predicted)
    i = [slice(None)] * fps.ndim
    i[-1] = -1
    fpr = fps.astype('float32') / np.expand_dims(fps[i], axis=-1)
    tpr = tps.astype('float32') / np.expand_dims(tps[i], axis=-1)
    #tpr = tps.astype('float32') / tps[i][:, np.newaxis]
    return fpr, tpr, thresholds

def last_axis_roc_auc_scores(y_true, y_predicted):
    fpr, tpr, _ = last_axis_roc_curve(y_true, y_predicted)
    return np.trapz(tpr, fpr)


def _matrix_clf_curve(y_true, y_predicted):
    assert y_true.ndim == y_predicted.ndim == 2
    row_i = T.arange(y_true.shape[0]).dimshuffle(0, 'x')
    col_i = y_predicted.argsort()
    reverse = [slice(None), slice(None, None, -1)]
    y_true = y_true[row_i, col_i][reverse]
    y_predicted = y_predicted[row_i, col_i][reverse]
    tps = y_true.cumsum(axis=-1)
    counts = T.ones_like(y_true) * T.arange(y_predicted.shape[-1])
    fps = 1 + counts - tps
    return fps, tps, y_predicted

def _tensor3_clf_curve(y_true, y_predicted):
    assert y_true.ndim == y_predicted.ndim == 3
    x_i = T.arange(y_true.shape[0]).dimshuffle(0, 'x', 'x')
    y_i = T.arange(y_true.shape[1]).dimshuffle('x', 0, 'x')
    z_i = y_predicted.argsort()
    reverse = [slice(None), slice(None), slice(None, None, -1)]
    y_true = y_true[x_i, y_i, z_i][reverse]
    y_predicted = y_predicted[x_i, y_i, z_i][reverse]
    tps = y_true.cumsum(axis=-1)
    counts = T.ones_like(y_true) * T.arange(y_predicted.shape[-1])
    fps = 1 + counts - tps
    return fps, tps, y_predicted

def _tensor4_clf_curve(y_true, y_predicted):
    assert y_true.ndim == y_predicted.ndim == 4
    a_i = T.arange(y_true.shape[0]).dimshuffle(0, 'x', 'x', 'x')
    b_i = T.arange(y_true.shape[1]).dimshuffle('x', 0, 'x', 'x')
    c_i = T.arange(y_true.shape[2]).dimshuffle('x', 'x', 0, 'x')
    d_i = y_predicted.argsort()

    reverse = [slice(None), slice(None), slice(None), slice(None, None, -1)]
    y_true = y_true[a_i, b_i, c_i, d_i][reverse]
    y_predicted = y_predicted[a_i, b_i, c_i, d_i][reverse]
    tps = y_true.cumsum(axis=-1)
    counts = T.ones_like(y_true) * T.arange(y_predicted.shape[-1])
    fps = 1 + counts - tps
    return fps, tps, y_predicted

def _binary_clf_curves(y_true, y_predicted):
    """
    returns curves calculated axis[-1]-wise

    note - despite trying several approaches, could not seem to get a
    n-dimensional verision of clf_curve to work, so abandoning. 2,3,4 is fine.

    """
    if not (y_true.ndim == y_predicted.ndim):
        raise ValueError('Dimension mismatch, ({}, {})'.format(y_true.ndim, y_predicted.ndim))
    if not isinstance(y_true, T.TensorVariable) or not isinstance(y_predicted, T.TensorVariable):
        raise TypeError('This only works for symbolic variables.')

    if y_true.ndim == 1:
        clf_curve_fn = _vector_clf_curve
    elif y_true.ndim == 2:
        clf_curve_fn = _matrix_clf_curve
    elif y_true.ndim == 3: 
        clf_curve_fn = _tensor3_clf_curve
    elif y_true.ndim == 4:
        clf_curve_fn = _tensor4_clf_curve
    else:
        raise NotImplementedError('Not implemented for ndim {}'.format(y_true.ndim))

    fps, tps, thresholds = clf_curve_fn(y_true, y_predicted)
    return fps, tps, thresholds

def _last_col_idx(ndim):
    last_col = [slice(None) for x in xrange(ndim)]
    last_col[-1] = -1
    return last_col

def _reverse_idx(ndim):
    reverse = [slice(None) for _ in range(ndim-1)]
    reverse.append(slice(None, None, -1))
    return reverse

def roc_curves(y_true, y_predicted):
    "returns roc curves calculated axis -1-wise"
    fps, tps, thresholds = _binary_clf_curves(y_true, y_predicted)
    last_col = _last_col_idx(y_true.ndim)
    fpr = fps.astype('float32') / T.shape_padright(fps[last_col], 1)
    tpr = tps.astype('float32') / T.shape_padright(tps[last_col], 1)
    return fpr, tpr, thresholds

def roc_auc_scores(y_true, y_predicted):
    "roc auc scores calculated axis -1-wise"
    fpr, tpr, thresholds = roc_curves(y_true, y_predicted)
    return auc(fpr, tpr)

def roc_auc_loss(y_true, y_predicted):
    return 1-roc_auc_scores(y_true, y_predicted)

def precision_recall_curves(y_true, y_predicted):
    "precision recall curves calculated axis -1-wise"
    fps, tps, thresholds = _binary_clf_curves(y_true, y_predicted)
    last_col = _last_col_idx(y_true.ndim)
    last_col[-1] = [-1] 
    precision = tps.astype('float32') / (tps + fps)
    if y_true.ndim == 1:
        recall = tps.astype('float32') / tps[-1]
    else:
        recall = tps.astype('float32') / tps[last_col]
    reverse = _reverse_idx(fps.ndim)
    precision = precision[reverse]
    recall = recall[reverse]
    thresholds = thresholds[reverse]
    if y_true.ndim == 1:
        ones, zeros = [1], [0]
    else:
        ones = T.ones_like(precision)[last_col]
        zeros = T.zeros_like(recall)[last_col]
    precision = T.concatenate([precision, ones], axis=-1) 
    recall = T.concatenate([recall, zeros], axis=-1)
    return precision, recall, thresholds

def average_precision_scores(y_true, y_predicted):
    precision, recall, _ = precision_recall_curves(y_true, y_predicted)
    return auc(recall, precision)

def precision_recall_loss(y_true, y_predicted):
    "convenience function to minimize for"
    return 1-average_precision_scores(y_true, y_predicted)



#aliases
roc_curve = roc_curves
roc_auc_score = roc_auc_scores
precision_recall_curve = precision_recall_curves
average_precision_score = average_precision_scores

def last_axis_precision_recall_curve(y_true, y_predicted):
    fps, tps, thresholds = _last_axis_binary_clf_curve(y_true, y_predicted)
    i = [slice(None)] * fps.ndim
    i[-1] = [-1]
    precision = tps.astype('float32') / (tps + fps)
    recall = tps.astype('float32') / tps[i]
    i[-1] = slice(None, None, -1)
    precision = precision[i]
    recall = recall[i]
    thresholds = thresholds[i]
    i[-1] = [-1]
    precision = np.concatenate([precision, np.ones(precision.shape)[i]], axis=-1)
    recall = np.concatenate([recall, np.zeros(recall.shape)[i]], axis=-1)
    return precision, recall, thresholds


