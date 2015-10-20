"""
A local/shared memory version of the reuse_kernel
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from numba import hsa, float32, float64
from functools import partial

from .utils import Benchmark, work_balanced_scaling

MAX_REUSE = 64
BLOCKSIZE = 4 * 64
CHUNKSIZE = BLOCKSIZE + MAX_REUSE


def make_kernel(typ):
    @hsa.jit
    def shared_reuse_kernel(dst, src, num_reuse):
        tid = hsa.get_local_id(0)
        tsz = hsa.get_local_size(0)
        gid = hsa.get_group_id(0)

        chunk = hsa.shared.array(CHUNKSIZE, dtype=typ)

        base = gid * tsz

        # Load chunk into shared memory
        chunk[tid] = src[base + tid]
        if tid < num_reuse:
            chunk[tsz + tid] = src[base + tsz + tid]

        hsa.barrier()

        # Actual work
        res = 0
        for j in range(num_reuse):
            res += chunk[tid + j]

        dst[base + tid] = res

    return shared_reuse_kernel


shared_reuse_kernel_f32 = make_kernel(float32)
shared_reuse_kernel_f64 = make_kernel(float64)


def launcher(timer, threads, blocks, size, dtype, num_reuse):
    assert size == threads * blocks
    src = np.random.random(size + num_reuse).astype(dtype)

    dst = np.zeros(size, dtype=src.dtype)
    kernel = {
        np.float32: shared_reuse_kernel_f32,
        np.float64: shared_reuse_kernel_f64,
    }[dtype]

    ts = timer()
    kernel[blocks, threads](dst, src, num_reuse)
    te = timer()

    return te - ts


bmlist = [Benchmark(name='sharedreuse{0}'.format(x),
                    launcher=partial(launcher, num_reuse=x),
                    scaling=partial(work_balanced_scaling, threads=BLOCKSIZE),
                    dtypes=[np.float32, np.float64])
          for x in [2, 4, 8, 16, 32, 64]]

if __name__ == '__main__':
    for bm in bmlist:
        bm.main()
