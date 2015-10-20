from __future__ import print_function, division, absolute_import
from timeit import default_timer
import pickle


class Benchmark(object):
    def __init__(self, name, launcher, scaling, dtypes, timer=default_timer,
                 repeat=3):
        self.name = name
        self.launcher = launcher
        self.scaling = scaling
        self.dtypes = dtypes
        self.timer = timer
        self.repeat = repeat

    def main(self):
        self.run()

    def run(self):
        print("== Running benchmark {0}".format(self.name))
        timings = []
        for dtype in self.dtypes:
            print("dtype", dtype)
            for threads, blocks, size in self.scaling():
                print("threads x blocks", threads, blocks)
                print("size", size)

                trials = []
                for _ in range(self.repeat):
                    duration = self.launcher(self.timer, threads, blocks, size,
                                             dtype)
                    trials.append(duration)

                print("durations", trials)

                data = dict(threads=threads,
                            blocks=blocks,
                            size=size,
                            dtype=dtype,
                            durations=trials)
                timings.append(data)

        # Save file
        with open("bm_data_{0}.pickle".format(self.name), 'wb') as fout:
            pickle.dump(timings, file=fout)


def work_balanced_scaling(maxblocks=2**16, threads_factor=8, wavesize=64,
                          maxsize=2 ** 30):

    blocks = 1
    threads = threads_factor * wavesize
    size = blocks * threads
    while size <= maxsize and blocks <= maxblocks:
        yield threads, blocks, size
        blocks *= 2
        size = blocks * threads
