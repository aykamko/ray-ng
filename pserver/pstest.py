import os
import ray

import numpy as np

from plasma import PlasmaClient

NUM_WORKERS = 5
NUM_SHARDS = NUM_WORKERS

_, addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)

client = PlasmaClient(addr_info['object_store_name'])

W = np.zeros((NUM_SHARDS, NUM_SHARDS), dtype=np.float64)
W_id = "x"*20

client.init_kvstore(W_id, W, shard_size=1)
result = client.pull(W_id, (0, 1))

def client_initializer():
    return PlasmaClient(addr_info['object_store_name'])
ray.reusables.local_client = ray.Reusable(client_initializer, lambda x: x)

@ray.remote
def worker(shard_idx):
    wpid = os.getpid()
    local_client = ray.reusables.local_client
    W_s = local_client.pull(W_id, (shard_idx, shard_idx+1))
    print('pid: {}\nshard_idx: {}\nshard: {}'.format(wpid, shard_idx, W_s))
    return 1

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

def run_worker(worker_idx):
    for i in range(1):
        update = ray.get(worker.remote(i))
        client.push(W_id, (i,i+1), client.pull(W_id, (i,i+1)) + update)

p = ThreadPool(NUM_WORKERS)
# p = Pool(NUM_WORKERS)
p.map(run_worker, range(NUM_WORKERS))
# p.join()
