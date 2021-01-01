import os
import multiprocessing as mp

NUM_CPUS = 3  # defaults to all available
py_file = 'train_dynamics'
params = [
    '--lr 1e-5',
    '--lr 1e-6',
    '--lr 1e-7',
]


def worker(i):
    os.system(f"python {py_file}.py {params[i]} --cuda 1 --worker 2 --l2 0 --emb-dim 64")


if __name__ == "__main__":
    with mp.Pool(NUM_CPUS) as pool:
        pool.map(worker, range(NUM_CPUS))
