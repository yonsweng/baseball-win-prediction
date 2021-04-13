import numpy as np
import random


def random_batch(n, batch_size=1):
    indice = list(range(n))
    random.shuffle(indice)
    for start_idx in range(0, n, batch_size):
        yield indice[start_idx:min(start_idx+batch_size, n)]


def unzip_batch(batch):
    parameters = []
    for i in range(len(batch[list(batch.keys())[0]])):
        parameters.append({column: batch[column][i] for column in batch})
    return parameters


def select_action(policy: np.array) -> int:
    return np.random.choice(len(policy), p=policy)
