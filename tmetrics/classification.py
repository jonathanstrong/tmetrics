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
multiclass_hinge_loss = lasagne.objectives.multiclass_hinge_loss
squared_error = lasagne.objectives.squared_error
binary_accuracy = lasagne.objectives.binary_accuracy
categorical_accuracy = lasagne.objectives.categorical_accuracy

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
        

