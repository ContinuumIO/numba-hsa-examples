from __future__ import print_function, division, absolute_import
import numpy as np
from numba import hsa
from functools import partial

from .utils import Benchmark, work_balanced_scaling


@hsa.jit
def mul_kernel(dst, src, n):
    i = hsa.get_global_id(0)
    out = 1
    for j in range(n):
        out *= src[i]
    dst[i] = out


def launcher(timer, threads, blocks, size, dtype, nop):
    assert size == threads * blocks
    src = np.ones(size, dtype=dtype)
    dst = np.zeros_like(src)
    ts = timer()
    mul_kernel[blocks, threads](dst, src, nop)
    te = timer()

    return te - ts


bm_mul1 = Benchmark(name='mul1',
                    launcher=partial(launcher, nop=1),
                    scaling=work_balanced_scaling,
                    dtypes=[np.float32, np.float64])

bm_mul32 = Benchmark(name='mul32',
                     launcher=partial(launcher, nop=32),
                     scaling=work_balanced_scaling,
                     dtypes=[np.float32, np.float64])

bm_mul64 = Benchmark(name='mul64',
                     launcher=partial(launcher, nop=64),
                     scaling=work_balanced_scaling,
                     dtypes=[np.float32, np.float64])

bm_mul128 = Benchmark(name='mul128',
                      launcher=partial(launcher, nop=128),
                      scaling=work_balanced_scaling,
                      dtypes=[np.float32, np.float64])

bm_mul256 = Benchmark(name='mul256',
                      launcher=partial(launcher, nop=256),
                      scaling=work_balanced_scaling,
                      dtypes=[np.float32, np.float64])

if __name__ == '__main__':
    bm_mul1.main()
    bm_mul32.main()
    bm_mul64.main()
    bm_mul128.main()
    bm_mul256.main()
