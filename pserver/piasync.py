import ray
import numpy as np
import lda.datasets
import time
from plasma import PlasmaClient

NUM_WORKERS = 10
NUM_ITER = 10000
# Start a scheduler, an object store, and some workers.
addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)

X = lda.datasets.load_reuters().astype(np.float64)
n_samples, n_features = X.shape

client = PlasmaClient(addr_info['object_store_name'])

X_id = "x"*20
X_shard_size = n_samples / NUM_WORKERS
client.init_kvstore(X_id, X, shard_order='C', shard_size=X_shard_size)

def worker_client_initializer():
    return PlasmaClient(addr_info['object_store_name'])
ray.reusables.local_client = ray.Reusable(worker_client_initializer, lambda x: x)

@ray.remote(num_return_vals=NUM_ITER)
def estimate_pi_yield(n, X_id, X_shard_idx, X_shard_size, n_feats):
  local_client = ray.reusables.local_client
  X_range = (X_shard_idx * X_shard_size,
             min((X_shard_idx + 1) * X_shard_size, n_feats))
  X_local = local_client.pull(X_id, X_range)

  for i in range(NUM_ITER):
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    yield 4 * np.mean(x ** 2 + y ** 2 < 1)

def pi_asynch(X_id, X_shard_size, n_feats):
    handle_matrix = np.empty((NUM_WORKERS, NUM_ITER), dtype=object)
    for i in range(NUM_WORKERS):
        iter_handles = estimate_pi_yield.remote(10000, X_id, i, X_shard_size, n_features)
        handle_matrix[i, :] = iter_handles
    return np.mean(ray.get([i for i in handle_matrix.flatten()]))

start = time.time()
e = pi_asynch(X_id, X_shard_size, n_features)
print "Pi is approximately {} in {}".format(e, time.time()-start)
