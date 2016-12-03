import time
import ray
import numpy as np

from plasma import PlasmaClient
from Queue import Queue
from sklearn.linear_model.logistic import _intercept_dot
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from svmlight_loader import load_svmlight_file
from scipy.special import expit
from scipy.sparse import issparse, csr_matrix
ray.register_class(csr_matrix)

# from sklearn.linear_model.sgd_fast import Log

from threading import Thread
from multiprocessing import Process
from multiprocessing.pool import ThreadPool

DEBUG = 1
def dprint(thing):
    if DEBUG:
        print(thing)

# Start ray
# NUM_WORKERS = 8
NUM_WORKERS = 5
NUM_GLOBAL_ITERS = 100
NUM_W_SHARDS = 10
NUM_WORKER_ITERS = NUM_GLOBAL_ITERS * NUM_W_SHARDS
addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)


# LR hyperparams
ALPHA = 1e-4
ETA = 0.02


# Source: https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/linear_model/logistic.py#L78
def logistic_grad(w, X, y, alpha):
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    # Logistic loss is the negative of the log of the logistic function.
    loss = -np.sum(log_logistic(yz)) + .5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()

    return loss, grad

def worker_client_initializer():
    return PlasmaClient(addr_info['object_store_name'])
ray.reusables.local_client = ray.Reusable(worker_client_initializer, lambda x: x)


@ray.remote(num_return_vals=NUM_WORKER_ITERS)
def async_compute_lr_grads(worker_idx, X_local, y_local, Wid, X_shard_idx, X_shard_size,
                           W_shard_size, num_w_shards, n_feats):

    dprint("started worker {}".format(worker_idx))
    local_client = ray.reusables.local_client

    for i in range(NUM_WORKER_ITERS):
        if i > 0:
            time.sleep(100)
        # TODO: tau-synchronization
        W_shard_idx = 0
        # W_shard_idx = np.random.randint(0, num_w_shards)
        W_range = (W_shard_idx*W_shard_size, min((W_shard_idx+1)*W_shard_size, n_feats))
        W_shard = local_client.pull(Wid, W_range) # TODO: reuse a numpy array here
        X_slice = X_local[:, range(*W_range)]

        _, grad = logistic_grad(W_shard, X_slice, y_local, ALPHA)
        dprint("worker {}: iter {}, computed grad for shard {}".format(worker_idx, i, W_shard_idx))
        yield (W_shard_idx, grad)


def driver_aggregate(worker_idx, W, Wid, worker_handles, W_shard_queues):
    dprint("started aggregate for worker {}".format(worker_idx))
    for i in range(NUM_WORKER_ITERS):
        time.sleep(1)
        result = ray.get(worker_handles[i])
        W_shard_idx, grad = result
        dprint("got grad for shard {} from worker {}".format(W_shard_idx, worker_idx))

        # W_shard_queues[W_shard_idx, worker_idx].put(grad)
        #
        # should_apply = not any([W_shard_queues[W_shard_idx, i].empty() for i in range(NUM_WORKERS)])
        # if not should_apply:
        #     continue
        #
        # dprint("accumed grad for shard {}".format(W_shard_idx))
        # accum_grad = np.zeros_like(grad)
        # for i in range(NUM_WORKERS):
        #     accum_grad += W_shard_queues[W_shard_idx, i].get()

        # TODO: apply gradient


def fit_async(X, y):
    client = PlasmaClient(addr_info['object_store_name'])

    n_samples, n_features = X.shape

    assert issparse(X), "X is not sparse"
    X_shard_size = n_samples / NUM_WORKERS

    Wid = "w"*20
    W = np.zeros((n_features,))
    W_shard_size = n_features / NUM_W_SHARDS
    client.init_kvstore(Wid, W, shard_order='F', shard_size=W_shard_size)

    W_shard_queues = np.empty((NUM_W_SHARDS+1, NUM_WORKERS), dtype=object)
    for i in range(NUM_W_SHARDS + 1):
        W_shard_queues[i, :] = [Queue() for _ in range(NUM_WORKERS)]

    handle_matrix = np.empty((NUM_WORKERS, NUM_WORKER_ITERS), dtype=object)
    for i in range(NUM_WORKERS):
      dprint("about to start worker {}".format(i))
      shard_range = (i*X_shard_size, min((i+1)*X_shard_size, n_features))
      y_shard = y[range(*shard_range)]
      X_shard = X[range(*shard_range)]
      iter_handles = async_compute_lr_grads.remote(
          i, X_shard, y_shard, Wid, i, X_shard_size, W_shard_size, NUM_W_SHARDS+1, # TODO: last arg not always true
          n_features)
      handle_matrix[i, :] = iter_handles

    def thread_driver_aggregate(i):
        driver_aggregate(i, W, Wid, handle_matrix[i], W_shard_queues)

    pool = ThreadPool(NUM_WORKERS)
    pool.map(thread_driver_aggregate, range(NUM_WORKERS))

    # def thread_driver_aggregate(worker_idx):
    #     return driver_aggregate(worker_idx, W, Wid, handle_matrix[worker_idx, :], W_shard_queues)
    # driver_pool.map(thread_driver_aggregate, range(NUM_WORKERS))

if __name__ == '__main__':
    def load_data():
        X_tr, y_tr = load_svmlight_file('data/spam1h.svm')
        return X_tr, y_tr
    X_tr_sparse, y_tr = load_data()
    # X_tr = np.array(X_tr_sparse.todense())  # XXX: huge

    dprint("starting async fit")
    fit_async(X_tr_sparse, y_tr)

    # n_samples, n_features = X_tr.shape
    # W = np.zeros((n_features,))
    #
    # eta = 0.02
    # alpha = 1e-4
    # logloss = Log()
    # for i in range(100):
    #     p = safe_sparse_dot(X_tr, W)
    #     loss, grad = logistic_grad(W, X_tr, y_tr, alpha)
    #     if i % 10 == 0:
    #         print("{} / 100 loss: {}".format(i, loss))
    #     W -= eta * grad
