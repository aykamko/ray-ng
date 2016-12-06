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
from multiprocessing.pool import ThreadPool

WDEBUG = 1
DEBUG = 1
def dprint(thing, level=3):
    if DEBUG >= level:
        print(thing)
def wprint(thing, level=3):
    if WDEBUG >= level:
        print(thing)

# Start ray
NUM_WORKERS = 4
MAJOR_ITERS = 10
# MINOR_ITERS = 5
MINOR_ITERS = 2
NUM_GLOBAL_ITERS = MAJOR_ITERS * MINOR_ITERS
TAU_DELAY = 2

NUM_W_SHARDS = 16
NUM_WORKER_ITERS = MINOR_ITERS * NUM_W_SHARDS

addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)


# LR hyperparams
ALPHA = 0.1
BETA = 1

LAMBDA1 = 3 # rub the crystal ball
LAMBDA2 = 0 # no L2 loss


def logistic_loss(w, X, y, alpha):
    w, c, yz = _intercept_dot(w, X, y)

    # Logistic loss is the negative of the log of the logistic function.
    return -np.sum(log_logistic(yz)) + .5 * alpha * np.dot(w, w)

# Source: https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/linear_model/logistic.py#L78
def logistic_grad(w, X, y, alpha):
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    z = expit(yz)
    z0 = (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()

    return grad

def worker_client_initializer():
    return PlasmaClient(addr_info['object_store_name'])
ray.reusables.local_client = ray.Reusable(worker_client_initializer, lambda x: x)


@ray.remote(num_return_vals=NUM_WORKER_ITERS)
def async_compute_lr_grads(worker_idx, X_local, y_local, Wid, tauid, X_shard_idx, X_shard_size,
                           W_shard_size, num_w_shards, n_feats):

    wprint("started worker {}".format(worker_idx))
    local_client = ray.reusables.local_client
    local_tau = np.zeros(num_w_shards)

    shards_left = range(num_w_shards)
    for i in range(NUM_WORKER_ITERS):
        if len(shards_left) == 0:
            shards_left = range(num_w_shards)
        wprint("worker {}: starting grad {}".format(worker_idx, i))

        while True:
            shards_left_idx = np.random.randint(len(shards_left))
            W_shard_idx = shards_left[shards_left_idx]
            shard_tau = int(local_client.pull(tauid, (W_shard_idx, W_shard_idx+1))[0])
            if shard_tau >= local_tau[W_shard_idx] - TAU_DELAY:
                shards_left.pop(shards_left_idx)
                local_tau[W_shard_idx] += 1
                break
            time.sleep(0.01)

        W_range = (W_shard_idx*W_shard_size, min((W_shard_idx+1)*W_shard_size, n_feats))
        W_shard = local_client.pull(Wid, W_range) # TODO: reuse a numpy array here
        X_slice = X_local[:, range(*W_range)]

        grad = logistic_grad(W_shard, X_slice, y_local, ALPHA)
        wprint("worker {}: computed grad {} for shard {}".format(worker_idx, i, W_shard_idx))
        yield (W_shard_idx, csr_matrix(grad))

    print("worker {} finished!".format(worker_idx))


# Some Block PG code from wormhole/ps-lite
# Source: https://github.com/dmlc/wormhole/blob/master/learn/linear

#  w.w = penalty.Solve(eta * w.w - grad[0], eta);

# inline T Solve(T z, T eta) {
#   // soft-thresholding
#   CHECK_GT(eta, 0);
#   if (z <= lambda1_ && z >= -lambda1_) return 0;
#   return (z > 0 ? z - lambda1_ : z + lambda1_) / (eta + lambda2_);
# }

def nzcols(X):
    return np.nonzero(X)[0]

def apply_grad(W, g, eta):
    z = eta*W - g
    z[(z >= -LAMBDA1) & (z <= LAMBDA1)] = 0
    z[z > 0] -= LAMBDA1
    z[z < 0] += LAMBDA1
    z /= eta + LAMBDA2
    return z

CONTAINS_RETRIES = 10000
def retry_get(client, handle):
    """jank"""
    tries = 0
    while True:
        time.sleep(0.1)
        if tries > CONTAINS_RETRIES:
            raise Exception('halp')
        if client.contains(handle.id()):
            break
        tries += 1
    return ray.get(handle)

TOTAL_RECIEVED = 0
def driver_aggregate(client, worker_idx, W, Wid, W_shard_size, n_feats,
                     worker_handles, W_shard_queues, W_shard_locks,
                     eta, total_t, tau, tauid):
    global TOTAL_RECIEVED

    dprint("started aggregate for worker {}".format(worker_idx))
    for i in range(NUM_WORKER_ITERS):
        handle = worker_handles[i]
        dprint("aggreg {} waiting for grad {} ({})".format(worker_idx, i, map(ord, handle.id()[:10])))
        result = retry_get(client, handle)
        TOTAL_RECIEVED += 1
        W_shard_idx, grad = result
        dprint("(TOTAL {}) aggreg {}, recieved grad {} for shard {}".format(
            TOTAL_RECIEVED, worker_idx, i, W_shard_idx))

        with W_shard_locks[W_shard_idx]:
            W_shard_queues[W_shard_idx, worker_idx].append((handle, grad))

            nonempty_slots = [len(W_shard_queues[W_shard_idx, j]) > 0 for j in range(NUM_WORKERS)]
            should_apply = all(nonempty_slots)
            dprint("aggreg {}, recieved grad for shard {}, grads recieved: {}".format(
                worker_idx, W_shard_idx, np.sum(nonempty_slots)))
            if not should_apply:
                continue

            accum_grad = np.zeros(grad.shape)
            for k in range(NUM_WORKERS):
                _, grad = W_shard_queues[W_shard_idx, k].pop()
                accum_grad += grad

            # necessary because of sparse gradients
            accum_grad = np.array(accum_grad)[0]

            W_range = (W_shard_idx*W_shard_size, min((W_shard_idx+1)*W_shard_size, n_feats))
            new_W_shard = apply_grad(W[range(*W_range)], accum_grad, eta)

            client.push(Wid, W_range, new_W_shard)
            tau[W_shard_idx] += 1
            client.push(tauid, (W_shard_idx, W_shard_idx+1), np.array(tau[W_shard_idx]))
            W[range(*W_range)] = new_W_shard

        total_t += 1
        eta = BETA + np.sqrt(total_t) / ALPHA

        dprint("applied grad for shard {}".format(W_shard_idx), level=1)

    print("aggregator {} finished!".format(worker_idx))

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

    def thread_driver_aggregate(i):
        driver_aggregate(client, i, W, Wid, W_shard_size, n_features,
                         handle_matrix[i], W_shard_queues, W_shard_locks,
                         eta, total_t, tau, tauid)

    total_t = 0
    eta = BETA + np.sqrt(total_t) / ALPHA

    pool = ThreadPool(NUM_WORKERS)

    X_shards, y_shards = [], []
    for i in range(NUM_WORKERS):
        shard_range = (i*X_shard_size, min((i+1)*X_shard_size, n_features))
        X_shards.append(X[range(*shard_range)])
        y_shards.append(y[range(*shard_range)])

    for mi in range(MAJOR_ITERS):
        print("mi {}, loss {}".format(mi, logistic_loss(W, X, y, ALPHA)))
        start_timer()
        for i in range(NUM_WORKERS):
            dprint("about to start worker {}".format(i))
            iter_handles = async_compute_lr_grads.remote(
                i, X_shards[i], y_shards[i], Wid, tauid, i, X_shard_size, W_shard_size,
                NUM_W_SHARDS, # TODO: not always true
                n_features)
            handle_matrix[i, :] = iter_handles

        pool.map(thread_driver_aggregate, range(NUM_WORKERS))
        print("finished major iter {}, time {} (total iters {})".format(
            mi+1, end_timer(), (mi+1)*MINOR_ITERS))
    print("mi {}, loss {}".format(mi, logistic_loss(W, X, y, ALPHA)))

T_START = 0
def start_timer():
    global T_START
    T_START = time.time()

def end_timer():
    global T_START
    return time.time() - T_START

from sklearn.externals.joblib import Memory
mem = Memory('ddp.cache')

@mem.cache
def load_data():
    X_tr, y_tr = load_svmlight_file('data/spam100k.svm')
    return X_tr, y_tr

if __name__ == '__main__':
    print("starting load")
    start_timer()
    X_tr_sparse, y_tr = load_data()
    print("finished load: {}".format(end_timer()))

    print("finished load, starting async fit")
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
