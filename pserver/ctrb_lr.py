import time
import sklearn.linear_model

import numpy as np

from sklearn.externals.joblib import Memory
from svmlight_loader.svmlight_loader import load_svmlight_file
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
mem = Memory('ctrb_lr.cache')

@mem.cache
def load_CTRb():
    X_tr, y_tr = load_svmlight_file('data/ctrb_train_1h.fixed', ind_dtype=np.uint64)
    X_te, y_te = load_svmlight_file('data/ctrb_test.fixed', n_features=X_tr.shape[1], ind_dtype=np.uint64)
    return X_tr, y_tr, X_te, y_te

X_tr, y_tr, X_te, y_te = load_CTRb()

import ipdb; ipdb.set_trace()
# X_coo = X_tr.tocoo()
W_tr = csr_matrix((X_tr.shape[1], 1))
#
# y_tr = X_tr.dot(W_tr)

# lrm = sklearn.linear_model.SGDRegressor()
#
# print("starting regression")
# t_start = time.time()
# lrm.fit(X_tr, y_tr)
# t_dur = time.time() - t_start
# print("finished regression! {} seconds".format(t_dur))
