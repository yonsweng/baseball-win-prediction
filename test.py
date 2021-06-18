import argparse
import numpy as np
from time import time
from utils import set_seeds


def load_test_args(parser):
    parser.add_argument('--test_batch_size', metavar='N', type=int,
                        default=1024,
                        help='the batch size for test')
    parser.add_argument('--seed', metavar='N', type=int, default=2021,
                        help='the random seed')
    return parser


def judge_winning_team(away_scores, home_scores):
    '''
    Return:
        list([who_won, ...])
        who_won:
            -1: away won, 0: draw, 1: home won
    '''
    return list(np.where(away_scores == home_scores, -1,
                np.where(away_scores > home_scores, -1, 1)))


def main():
    parser = argparse.ArgumentParser(description='Test argument parser')
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    start_time = time()
    print(f'test time: {time() - start_time:.3f} sec.')


if __name__ == '__main__':
    main()
