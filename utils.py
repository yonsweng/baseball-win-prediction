import numpy as np


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def unzip_batch(batch):
    parameters = []
    for i in range(len(batch[list(batch.keys())[0]])):
        parameters.append({column: batch[column][i] for column in batch})
    return parameters


def select_action(policy: np.array) -> int:
    return np.random.choice(len(policy), p=policy)
