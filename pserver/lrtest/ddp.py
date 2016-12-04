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

from threading import Thread, RLock
from multiprocessing import Process
from multiprocessing.pool import ThreadPool

DEBUG = 1
def dprint(thing):
    if DEBUG:
        print(thing)

# Start ray
NUM_WORKERS = 8
NUM_GLOBAL_ITERS = 100
NUM_W_SHARDS = 5
NUM_WORKER_ITERS = NUM_GLOBAL_ITERS * NUM_W_SHARDS
addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)


# LR hyperparams
ALPHA = 0.1
BETA = 1
ETA = 0.1

LAMBDA1 = 4
LAMBDA2 = 1e-4


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


# TAU_DELAY = 8
TAU_DELAY = 1
@ray.remote(num_return_vals=NUM_WORKER_ITERS)
def async_compute_lr_grads(worker_idx, X_local, y_local, Wid, tauid, X_shard_idx, X_shard_size,
                           W_shard_size, num_w_shards, n_feats):

    dprint("started worker {}".format(worker_idx))
    local_client = ray.reusables.local_client
    local_tau = np.zeros(num_w_shards)

    shards_left = range(num_w_shards)
    for i in range(NUM_WORKER_ITERS):
        if len(shards_left) == 0:
            shards_left = range(num_w_shards)

        while True:
            rnd_idx = np.random.randint(len(shards_left))
            W_shard_idx = shards_left[rnd_idx]
            shard_tau = int(local_client.pull(tauid, (W_shard_idx, W_shard_idx+1))[0])
            if shard_tau >= local_tau[shard_tau] - TAU_DELAY:
                shards_left.pop(rnd_idx)
                local_tau[W_shard_idx] += 1
                break
            time.sleep(0.01)
            # dprint("delayed on block {}".format(W_shard_idx))

        W_range = (W_shard_idx*W_shard_size, min((W_shard_idx+1)*W_shard_size, n_feats))
        W_shard = local_client.pull(Wid, W_range) # TODO: reuse a numpy array here
        X_slice = X_local[:, range(*W_range)]

        _, grad = logistic_grad(W_shard, X_slice, y_local, ALPHA)
        # dprint("worker {}: iter {}, computed grad for shard {}".format(worker_idx, i, W_shard_idx))
        yield (W_shard_idx, grad)


# Some block PG code from ps-lite
# Source: https://github.com/dmlc/wormhole/blob/master/learn/linear

#  w.w = penalty.Solve(eta * w.w - grad[0], eta);

# inline T Solve(T z, T eta) {
#   // soft-thresholding
#   CHECK_GT(eta, 0);
#   if (z <= lambda1_ && z >= -lambda1_) return 0;
#   return (z > 0 ? z - lambda1_ : z + lambda1_) / (eta + lambda2_);
# }

def apply_grad(W, g, eta):
    z = eta*W - g
    z[z <= LAMBDA1] = 0
    z[z <= -LAMBDA1] = 0
    z[z > 0] -= LAMBDA1
    z[z < 0] += LAMBDA1
    z /= eta + LAMBDA2
    return z

TOTAL_RECIEVED = 0

FOREVER = 999999
def driver_aggregate(client, worker_idx, W, Wid, W_shard_size, n_feats,
                     worker_handles, W_shard_queues, W_shard_locks,
                     eta, total_t, tau, tauid):
    global TOTAL_RECIEVED

    dprint("started aggregate for worker {}".format(worker_idx))
    for i in range(NUM_WORKER_ITERS):
        dprint("waiting for grad {} from {}".format(i, worker_idx))
        ray.wait([worker_handles[i]], timeout=FOREVER)
        result = ray.get(worker_handles[i])
        TOTAL_RECIEVED += 1
        W_shard_idx, grad = result
        dprint("(total {}) aggreg {}, recieved grad for shard {}".format(
            TOTAL_RECIEVED, worker_idx, W_shard_idx))

        with W_shard_locks[W_shard_idx]:
            W_shard_queues[W_shard_idx, worker_idx].append(grad)

            nonempty_slots = [len(W_shard_queues[W_shard_idx, j]) > 0 for j in range(NUM_WORKERS)]
            should_apply = all(nonempty_slots)
            dprint("aggreg {}, recieved grad for shard {}, grads recieved: {}".format(
                worker_idx, W_shard_idx, np.sum(nonempty_slots)))
            # dprint("nonempty_slots: {}".format(nonempty_slots))
            if not should_apply:
                continue

            # import ipdb; ipdb.set_trace()
            accum_grad = np.zeros_like(grad)
            for k in range(NUM_WORKERS):
                accum_grad += W_shard_queues[W_shard_idx, k].pop()

        W_range = (W_shard_idx*W_shard_size, min((W_shard_idx+1)*W_shard_size, n_feats))
        new_W_shard = apply_grad(W[range(*W_range)], accum_grad, eta)

        # import ipdb; ipdb.set_trace()
        print(W_shard_idx, new_W_shard.shape, W_range)
        client.push(Wid, W_range, new_W_shard)
        tau[W_shard_idx] += 1
        client.push(tauid, (W_shard_idx, W_shard_idx+1), np.array(tau[W_shard_idx]))
        W[range(*W_range)] = new_W_shard

        total_t += 1
        eta = BETA + np.sqrt(total_t) / ALPHA

        dprint("applied grad for shard {}".format(W_shard_idx))


def fit_async(X, y):
    client = PlasmaClient(addr_info['object_store_name'])

    n_samples, n_features = X.shape

    assert issparse(X), "X is not sparse"
    X_shard_size = n_samples / NUM_WORKERS

    Wid = "w"*20
    W = np.zeros((n_features,))
    W_shard_size = n_features / NUM_W_SHARDS
    client.init_kvstore(Wid, W, shard_order='F', shard_size=W_shard_size)

    tauid = "t"*20
    tau = np.zeros(NUM_W_SHARDS)
    client.init_kvstore(tauid, tau, shard_size=1)

    W_shard_locks = np.empty((NUM_W_SHARDS,), dtype=object)
    W_shard_queues = np.empty((NUM_W_SHARDS, NUM_WORKERS), dtype=object)
    for i in range(NUM_W_SHARDS):
        W_shard_locks[i] = RLock()
        W_shard_queues[i, :] = [[] for _ in range(NUM_WORKERS)]

    handle_matrix = np.empty((NUM_WORKERS, NUM_WORKER_ITERS), dtype=object)
    for i in range(NUM_WORKERS):
      dprint("about to start worker {}".format(i))
      shard_range = (i*X_shard_size, min((i+1)*X_shard_size, n_features))
      y_shard = y[range(*shard_range)]
      X_shard = X[range(*shard_range)]
      iter_handles = async_compute_lr_grads.remote(
          i, X_shard, y_shard, Wid, tauid, i, X_shard_size, W_shard_size,
          NUM_W_SHARDS, # TODO: not always true
          n_features)
      handle_matrix[i, :] = iter_handles

    total_t = 0
    eta = BETA + np.sqrt(total_t) / ALPHA

    def thread_driver_aggregate(i):
        driver_aggregate(client, i, W, Wid, W_shard_size, n_features,
                         handle_matrix[i], W_shard_queues, W_shard_locks,
                         eta, total_t, tau, tauid)

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

    # HACK
    _, n_feats = X_tr_sparse.shape
    left_over = n_feats % NUM_W_SHARDS
    X_tr_sparse = X_tr_sparse[:, :-left_over]

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
