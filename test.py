import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from time import time
from ActionSpace import ActionSpace
from utils import sequential_batch, to_batch_input, set_seeds, get_test_data, \
                  create_nnet


def load_test_args(parser):
    parser.add_argument('--test_size', metavar='N', type=int, default=100,
                        help='the number of tests')
    parser.add_argument('--test_batch_size', metavar='N', type=int,
                        default=4096,
                        help='the number of episodes to simulate for an epoch')
    parser.add_argument('--seed', metavar='N', type=int, default=2021,
                        help='the random seed')
    parser.add_argument('--n_actions', metavar='N', type=int,
                        default=len(ActionSpace()),
                        help='the number of actions')
    parser.add_argument('--n_gpus', metavar='N', type=int,
                        default=torch.cuda.device_count(),
                        help='the number of GPUs to use')
    return parser


def judge_winning_team(away_scores, home_scores):
    '''
    Return:
        list([who_won, ...])
        who_won:
            -1: away won, 0: draw, 1: home won
    '''
    if torch.is_tensor(away_scores):
        away_scores = away_scores.cpu().numpy()
    if torch.is_tensor(home_scores):
        home_scores = home_scores.cpu().numpy()
    return list(np.where(away_scores == home_scores, 0,
                np.where(away_scores > home_scores, -1, 1)))


def test(data, nnet, device, args):
    y_true, y_pred = [], []

    for indice in sequential_batch(len(data), args.test_batch_size):
        batch = data[indice]
        input = to_batch_input(batch, device)
        with torch.no_grad():
            _, value = nnet(input)
        y_true += judge_winning_team(batch['AWAY_END_SCORE_CT'],
                                     batch['HOME_END_SCORE_CT'])
        y_pred += judge_winning_team(value[:, 0], value[:, 1])

    accuracy = accuracy_score(y_true, y_pred)
    print(f'accuracy: {accuracy:.3f}')


def main():
    parser = argparse.ArgumentParser(description='Test argument parser')
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    test_data = get_test_data()

    device = torch.device('cuda:0')

    nnet = create_nnet(test_data, args)

    start_time = time()
    test(test_data, nnet, device, args)
    print(f'test time: {time() - start_time:.3f} sec.')


if __name__ == '__main__':
    main()
