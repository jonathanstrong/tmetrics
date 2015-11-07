import theano, theano.tensor as T
import numpy as np
import pandas as pd
import lasagne

#copy existing code and place in tmetrics namespace
binary_crossentropy = T.nnet.binary_crossentropy
categorical_crossentropy = T.nnet.categorical_crossentropy
binary_hinge_loss = lasagne.objectives.binary_hinge_loss
multiclass_hinge_loss = lasagne.objectives.multiclass_hinge_loss
squared_error = lasagne.objectives.squared_error
binary_accuracy = lasagne.objectives.binary_accuracy
categorical_accuracy = lasagne.objectives.categorical_accuracy

def brier_score_loss(y_true, y_predicted, sample_weight=None):
    """
    port of sklearn.metrics.brier_score_loss
    works for 2D binary data as well, e.g.

    [[0, 1, 0],
     [1, 0, 0]]

    [[.1, .2, .3],
     [.4, .5, .6]]

    y_true: tensor 
    y_predicted: tensor  
    sample_weight: tensor or None (standard mean)

    assumptions: 
     -binary ground truth values ({0, 1})
     -probabilities are floats between 0-1
     -sample_weight broadcasts to ((y_true - y_predicted) ** 2)

    """
    scores = ((y_true - y_predicted) ** 2)
    if sample_weight is not None: 
        scores = scores * sample_weight
    return scores.mean()



