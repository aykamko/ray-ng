import ray
import time

ray.init(start_ray_local=True, num_workers=1)

@ray.remote(num_return_vals=10)
def foo():
    for i in range(10):
        yield i

handles = foo.remote()
for i, h in enumerate(handles):
    result = ray.get(h)
    print(i, result)
