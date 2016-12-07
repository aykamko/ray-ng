import ray
import numpy as np
import lda.datasets
import time
from plasma import PlasmaClient

NUM_WORKERS = 10
NUM_ITER = 1000
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

# Define a remote function for estimating pi.
@ray.remote
def estimate_pi(n, X_id, X_shard_idx, X_shard_size, n_feats):
  # local_client = ray.reusables.local_client
  # X_range = (X_shard_idx * X_shard_size,
  #            min((X_shard_idx + 1) * X_shard_size, n_feats))
  # X_local = local_client.pull(X_id, X_range)
  x = np.random.uniform(size=n)
  y = np.random.uniform(size=n)
  return 4 * np.mean(x ** 2 + y ** 2 < 1)

# Launch 10 tasks, each of which estimates pi.
start = time.time()
result_ids = []
estimate = 0
for j in range(NUM_ITER):
    for _ in range(NUM_WORKERS):
      result_ids.append(estimate_pi.remote(10000, X_id, _, X_shard_size, n_features))

# Fetch the results of the tasks and print their average.
estimate = np.mean(ray.get(result_ids))
print "Pi is approximately {} in {}".format(estimate, time.time()-start)
