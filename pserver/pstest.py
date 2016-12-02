import time
import os
import ray

import numpy as np

from plasma import PlasmaClient

WORKER_SLEEP = 0.01
NUM_WORKERS = 5
NUM_SHARDS = NUM_WORKERS
NUM_ITERS = 5

addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)

client = PlasmaClient(addr_info['object_store_name'])

W = np.zeros((NUM_SHARDS,), dtype=np.float64)
W_id = "x"*20

client.init_kvstore(W_id, W, shard_size=1)

def client_initializer():
    return PlasmaClient(addr_info['object_store_name'])
ray.reusables.local_client = ray.Reusable(client_initializer, lambda x: x)

@ray.remote(num_return_vals=NUM_ITERS)
def worker(shard_idx, num_iters):
    local_client = ray.reusables.local_client
    for i in range(num_iters):
        time.sleep(WORKER_SLEEP)
        W_s = local_client.pull(W_id, (shard_idx, shard_idx+1))
        yield W_s + 1
        # yield 1

handle_matrix = np.empty((NUM_WORKERS, NUM_ITERS), dtype=object)
for i in range(NUM_WORKERS):
    handle_matrix[i, :] = worker.remote(i, NUM_ITERS)

for i in range(NUM_ITERS):
    results = np.zeros_like(W)
    for j in range(NUM_WORKERS):
        results[j] = ray.get(handle_matrix[j, i])
    client.push(W_id, (0, NUM_SHARDS), results)

# results = np.zeros_like(W)
# for i in range(NUM_ITERS):
#     for j in range(NUM_WORKERS):
#         results[j] += ray.get(handle_matrix[j, i])

print(results)
