"""
Test performance of caching behavior in a chunked algorithm with
chunks of each thread overlaps.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from numba import hsa
from functools import partial

from .utils import Benchmark, work_balanced_scaling


@hsa.jit
def reuse_kernel(dst, src, num_reuse):
    i = hsa.get_global_id(0)

    res = 0
    for j in range(num_reuse):
        res += src[i + j]

    dst[i] = res


def launcher(timer, threads, blocks, size, dtype, num_reuse):
    assert size == threads * blocks
    src = np.random.random(size + num_reuse).astype(dtype)

    dst = np.zeros(size, dtype=src.dtype)

    ts = timer()
    reuse_kernel[blocks, threads](dst, src, num_reuse)
    te = timer()

    return te - ts


bmlist = [Benchmark(name='reuse{0}'.format(x),
                    launcher=partial(launcher, num_reuse=x),
                    scaling=work_balanced_scaling,
                    dtypes=[np.float32, np.float64])
          for x in [2, 4, 8, 16, 32]]

if __name__ == '__main__':
    for bm in bmlist:
        bm.main()
