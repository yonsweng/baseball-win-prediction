import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from time import time
from ActionSpace import ActionSpace
from utils import sequential_dataset_batch, to_input_batch, set_seeds, \
                  get_test_data, create_nnet


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
    parser.add_argument('--load', metavar='S', type=str,
                        default='pretrained.pt', help='a model to load')
    return parser


def judge_winning_team(away_scores, home_scores):
    '''
    Return:
        list([who_won, ...])
        who_won:
            -1: away won, 0: draw, 1: home won
    '''
    return list(np.where(away_scores == home_scores, 0,
                np.where(away_scores > home_scores, -1, 1)))


def test(data, nnet, args, tb, epoch, device=torch.device('cuda:0')):
    y_true, y_pred = [], []

    for batch in sequential_dataset_batch(data, args.test_batch_size):
        input = to_input_batch(batch, device)

        with torch.no_grad():
            _, value = nnet(input)

        y_true += judge_winning_team(batch['AWAY_END_SCORE_CT'],
                                     batch['HOME_END_SCORE_CT'])

        away_pred_scores = value[:, 0].cpu().numpy() + batch['AWAY_SCORE_CT']
        home_pred_scores = value[:, 1].cpu().numpy() + batch['HOME_SCORE_CT']
        y_pred += judge_winning_team(away_pred_scores, home_pred_scores)

    accuracy = accuracy_score(y_true, y_pred)
    print(f'accuracy: {accuracy:.3f}')
    tb.add_scalar('accuracy', accuracy, epoch)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test argument parser')
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    test_data = get_test_data()

    nnet = create_nnet(test_data, args)

    nnet.module.load_state_dict(torch.load(f'models/{args.load}'))

    tb = SummaryWriter()

    start_time = time()
    test(test_data, nnet, args, tb, epoch=0)
    print(f'test time: {time() - start_time:.3f} sec.')


if __name__ == '__main__':
    main()
