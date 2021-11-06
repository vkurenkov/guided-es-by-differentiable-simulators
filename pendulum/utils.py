import numpy as np

class NormalizationBuffer:
  def __init__(self, size):
    self._replay   = [0] * size
    self._size     = size

    self._cur_size = 0
    self._index    = 0

  def reset(self):
      self._replay   = [0] * self._size

  def append(self, memento):
    self._replay[self._index] = memento
    self._index    = (self._index + 1) % self._size
    self._cur_size = min(self._cur_size + 1, self._size)

  def mean_std(self):
    vals = []
    for i in range(self._cur_size):
        vals.append(self._replay[i])

    return np.mean(vals), np.std(vals)

class RunningStat:
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count