import argparse
from BaseballDataset import get_test_data


def load_test_args(parser):
    parser.add_argument('--test_size', metavar='N', type=int, default=100,
                        help='the number of tests')
    parser.add_argument('--test_batch_size', metavar='N', type=int,
                        default=1024,
                        help='the number of episodes to simulate for an epoch')
    return parser


def test(data, args):
    pass


def main():
    parser = argparse.ArgumentParser(description='Test argument parser')
    parser = load_test_args(parser)
    args = parser.parse_args()

    test_data = get_test_data(args)

    test(test_data, args)


if __name__ == '__main__':
    main()
