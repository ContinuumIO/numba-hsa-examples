"""
Use a copy kernel to determine the maximum throughput.
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from numba import hsa

from .utils import Benchmark, work_balanced_scaling


@hsa.jit
def copy_kernel(dst, src):
    i = hsa.get_global_id(0)
    dst[i] = src[i]


def launcher(timer, threads, blocks, size, dtype):
    assert size == threads * blocks
    src = np.arange(size, dtype=dtype)
    dst = np.zeros_like(src)
    ts = timer()
    copy_kernel[blocks, threads](dst, src)
    te = timer()
    # check result
    np.testing.assert_equal(dst, src)
    return te - ts


benchmark = Benchmark(name='copy',
                      launcher=launcher,
                      scaling=work_balanced_scaling,
                      dtypes=[np.float32, np.float64])

if __name__ == '__main__':
    benchmark.main()
