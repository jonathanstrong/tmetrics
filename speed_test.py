import tmetrics
from tabulate import tabulate
import time
import numpy as np
import theano
import theano.tensor as T
import sklearn.metrics

def _test(fn, yt, yp, n_iter=3):
    times = []
    for _ in xrange(n_iter):
        start = time.time()
        result = fn(yt, yp)
        times.append(time.time() - start)
    return np.mean(times)



def speed_test_roc_auc_four_dims():
    shapes = [(5000,), (5000, 100), (500, 300, 40), (500, 4, 500, 500)]
    types = [T.fvector, T.fmatrix, T.ftensor3, T.ftensor4]
    results = []
    for shape, _type, in zip(shapes, types):
        ndim = len(shape)
        yt = _type('yt')
        yp = _type('yp')
        gpu_roc_auc_expr = tmetrics.roc_auc_scores(yt, yp)
        roc_auc = theano.function([yt, yp], gpu_roc_auc_expr)

        yt_val = np.random.binomial(n=1, p=.5, size=shape).astype('float32')
        yp_val = np.random.random(shape).astype('float32')

        tmetrics_time = _test(roc_auc, yt_val, yp_val)
        if ndim == 1: 
            sklearn_time = _test(sklearn.metrics.roc_auc_score, yt_val, yp_val)
        else:
            sklearn_time = np.nan
        numpy_time = _test(tmetrics.last_axis_roc_auc_scores, yt_val, yp_val)

        results.append((ndim, numpy_time, tmetrics_time, sklearn_time))
    print tabulate(results, headers=['ndim', 'numpy', 'tmetrics', 'sklearn (vector only)'])

if __name__ == '__main__':
    speed_test_roc_auc_four_dims()



