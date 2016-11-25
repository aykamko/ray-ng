import numpy as np

from sklearn.externals.joblib import Memory
from svmlight_loader.svmlight_loader import load_svmlight_file
mem = Memory('ctrb_lr.cache')

# @mem.cache
def load_CTRb():
    X_tr, y_tr = load_svmlight_file('ctrb_train_1h')
    X_te, y_te = load_svmlight_file('ctrb_test')
    return X_tr, y_tr, X_te, y_te

X_tr, y_tr, X_te, y_te = load_CTRb()

import sklearn.linear_model

lrm = sklearn.linear_model.SGDRegressor()
