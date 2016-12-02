import ray
import time

ray.init(start_ray_local=True, num_workers=1)

@ray.remote
def foo():
    for i in range(10):
        yield i

handle = foo.remote()
time.sleep(100)
print 'Done sleeping!'
result = ray.get(handle)
print result
