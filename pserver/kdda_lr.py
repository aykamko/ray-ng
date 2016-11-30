import time
import sklearn.linear_model

import numpy as np

from sklearn.externals.joblib import Memory
from svmlight_loader.svmlight_loader import load_svmlight_file
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
mem = Memory('ctrb_lr.cache')

@mem.cache
def load_CTRb():
    X_tr, y_tr = load_svmlight_file('data/kdda')
    X_te, y_te = load_svmlight_file('data/kdda.t', n_features=X_tr.shape[1])
    return X_tr, y_tr, X_te, y_te

X_tr, y_tr, X_te, y_te = load_CTRb()

W_tr = csr_matrix((X_tr.shape[1], 1))

import ipdb; ipdb.set_trace()
y_tr = X_tr.dot(W_tr)

# lrm = sklearn.linear_model.SGDRegressor()
#
# print("starting regression")
# t_start = time.time()
# lrm.fit(X_tr, y_tr)
# t_dur = time.time() - t_start
# print("finished regression! {} seconds".format(t_dur))
