import argparse
from collections import deque
from functools import partial
import torch
import torch.multiprocessing as mp
from BaseballDataset import get_train_data, get_valid_data
from Env import Env
from MCTS import MCTS
from utils import random_batch, unzip_batch, select_action
from test import load_test_args, test


def load_train_args(parser):
    parser.add_argument('--examples_len', metavar='N', type=int,
                        default=100000,
                        help='the maximum length of train examples')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=16,
                        help='the number of episodes to simulate for an epoch')
    parser.add_argument('--n_cpus', metavar='N', type=int,
                        default=16,
                        help='the number of processors to use')
    parser.add_argument('--n_gpus', metavar='N', type=int,
                        default=torch.cuda.device_count(),
                        help='the number of GPUs to use')
    return parser


def make_an_episode(rank, state, args):
    examples = []
    tmp_examples = []
    curr_scores = [0, 0]
    env = Env()
    mcts = MCTS(rank, args)

    state = env.reset(state)
    while True:
        policy = mcts.get_policy(state)
        action = select_action(policy)
        state, runs_scored, done, _ = env.step(action)
        curr_scores[0] += runs_scored[0]
        curr_scores[1] += runs_scored[1]
        tmp_examples.append((state, policy, tuple(curr_scores)))
        if done:
            break

    for example in tmp_examples:
        future_runs = (curr_scores[0] - example[2][0],
                       curr_scores[1] - example[2][1])
        examples.append((example[0], example[1], future_runs))

    return examples


def simulate(batch, args):
    with mp.Pool(args.n_cpus) as pool:
        func = partial(make_an_episode, args=args)
        parameters = unzip_batch(batch)
        examples_list = pool.starmap(func, enumerate(parameters))
    return [example for examples in examples_list for example in examples]


def update_net(train_examples):
    pass


def main():
    parser = argparse.ArgumentParser(description='Training argument parser')
    parser = load_train_args(parser)
    parser = load_test_args(parser)
    args = parser.parse_args()

    train_data = get_train_data()
    valid_data = get_valid_data()

    train_examples = deque(maxlen=args.examples_len)

    epoch = 0
    while True:
        for indice in random_batch(len(train_data), args.train_batch_size):
            epoch += 1
            print(f'Epoch {epoch}')

            train_examples.extend(simulate(train_data[indice], args))
            update_net(train_examples)
            test(valid_data, args)


if __name__ == '__main__':
    main()
