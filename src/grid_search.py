import os
import multiprocessing as mp

NUM_CPUS = 4  # defaults to all available
py_file = 'train_prediction'
params = [
    '--lr 3e-4 --l2 0.1',
    '--lr 1e-4 --l2 0.1',
    '--lr 3e-4 --l2 0.01',
    '--lr 1e-4 --l2 0.01'
]


def worker(i):
    os.system(f"python {py_file}.py {params[i]} --cuda 1 --worker 4 --batch-size 2048 --patience 10")


if __name__ == "__main__":
    with mp.Pool(NUM_CPUS) as pool:
        pool.map(worker, range(NUM_CPUS))
