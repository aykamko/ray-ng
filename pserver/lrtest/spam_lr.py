import time
import sklearn.linear_model

import numpy as np

from sklearn.linear_model.logistic import _intercept_dot
from sklearn.externals.joblib import Memory
from svmlight_loader import load_svmlight_file
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.special import expit
from sklearn.utils.extmath import log_logistic, safe_sparse_dot

NUM_ITERS = 1000
ALPHA = 1e-4
BETA = 1

def logistic_loss(w, X, y, alpha):
    w, c, yz = _intercept_dot(w, X, y)

    # Logistic loss is the negative of the log of the logistic function.
    return -np.sum(log_logistic(yz)) + .5 * alpha * np.dot(w, w)

def logistic_grad(w, X, y, alpha):
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    z = expit(yz)
    z0 = (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0)

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()

    return grad

from sklearn.externals.joblib import Memory
mem = Memory('ddp.cache')

@mem.cache
def load_data():
    X_tr, y_tr = load_svmlight_file('data/spam100k.svm')
    return X_tr, y_tr

print("starting load...")
t_start = time.time()
X_tr, y_tr = load_data()
t_dur = time.time() - t_start
print("load duration: {} s".format(t_dur))

total_t = 0

n_samples, n_features = X_tr.shape
W = np.zeros((n_features,))
eta = BETA + np.sqrt(total_t) / ALPHA

print("loss {}".format(logistic_loss(W, X_tr, y_tr, 0)))
t_start = time.time()
for i in range(NUM_ITERS):
    g = logistic_grad(W, X_tr, y_tr, 0)
    W -= eta*g

    total_t += 1
    eta = BETA + np.sqrt(total_t) / ALPHA

    if i % 10 == 0:
        print('finished iter {}'.format(i))

t_dur = time.time() - t_start
print("fit duration: {} s".format(t_dur))
print("loss {}".format(logistic_loss(W, X_tr, y_tr, 0)))

# sklearn
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
