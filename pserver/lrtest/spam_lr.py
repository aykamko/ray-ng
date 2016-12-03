import time
import sklearn.linear_model

import numpy as np

from sklearn.externals.joblib import Memory
from svmlight_loader import load_svmlight_file
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

def load_data():
    X_tr, y_tr = load_svmlight_file('data/spam100k.svm')
    # X_tr, y_tr = load_svmlight_file('data/spam1h.svm')
    return X_tr, y_tr

print("starting load...")
t_start = time.time()
X_tr, y_tr = load_data()
t_dur = time.time() - t_start
print("load duration: {} s".format(t_dur))

lrm = sklearn.linear_model.SGDClassifier(
    penalty='l1', n_jobs=8,
    verbose=1,
)

t_start = time.time()
lrm.fit(X_tr, y_tr)
t_dur = time.time() - t_start

print("fit duration: {} s".format(t_dur))

# starting load...
# load duration: 571.475234032 s
# -- Epoch 1
# Norm: 116.86, NNZs: 392, Bias: -0.958352, T: 100000, Avg. loss: 0.108893
# Total training time: 6.05 seconds.
# -- Epoch 2
# Norm: 120.49, NNZs: 370, Bias: -0.969282, T: 200000, Avg. loss: 0.101853
# Total training time: 12.12 seconds.
# -- Epoch 3
# Norm: 122.51, NNZs: 362, Bias: -0.972334, T: 300000, Avg. loss: 0.099110
# Total training time: 18.25 seconds.
# -- Epoch 4
# Norm: 123.93, NNZs: 352, Bias: -0.979018, T: 400000, Avg. loss: 0.097535
# Total training time: 24.31 seconds.
# -- Epoch 5
# Norm: 125.01, NNZs: 348, Bias: -0.975650, T: 500000, Avg. loss: 0.096503
# Total training time: 30.38 seconds.
# fit duration: 35.3635690212 s
