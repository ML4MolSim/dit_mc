import collections
import itertools
import threading
import queue
import jax
import jax.numpy as jnp
import numpy as np

class PrefetchBuffer:
    def __init__(self, data_generator, buffer_size=2, device=None):
        self.data_generator = data_generator
        self.buffer_size = buffer_size
        self.device = device if device else jax.devices()[0]
        
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_signal = object()
        self.thread = threading.Thread(target=self._fill_buffer, daemon=True)
        self.thread.start()

    def _fill_buffer(self):
        for data in self.data_generator:
            # Transfer to device if necessary
            data = jax.device_put(data, self.device)
            self.queue.put(data)
        self.queue.put(self.stop_signal)

    def __next__(self):
        data = self.queue.get()
        if data is self.stop_signal:
            raise StopIteration
        return data
    
    def __iter__(self):
        return self  # This makes it an iterator


# This is a version of prefetch_to_device
# that works with only one GPU
def prefetch_single(iterator, size):
    queue = collections.deque()
    device = jax.devices()[0]

    def _prefetch(xs):
        return jax.device_put(xs, device)

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)
