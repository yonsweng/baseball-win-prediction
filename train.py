import argparse
from collections import deque
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import Adam
from Env import Env
from ActionSpace import ActionSpace
from MCTS import MCTS
from test import load_test_args, test
from utils import select_action, sequential_batch, set_seeds, copy_nnet, \
                  create_nnet, create_nnets, get_train_data, get_valid_data, \
                  random_batch


def load_train_args(parser):
    parser.add_argument('--lr', metavar='F', type=float,
                        default=1e-4, help='the learning rate')
    parser.add_argument('--examples_len', metavar='N', type=int,
                        default=100000,
                        help='the maximum length of train examples')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=512,
                        help='the number of episodes to simulate for an epoch')
    parser.add_argument('--n_cpus', metavar='N', type=int,
                        default=15,
                        help='the number of processors to use')

    # MCTS arguments
    parser.add_argument('--num_mcts_sims', metavar='N', type=int,
                        default=500, help='the number of MCTS simulations')
    parser.add_argument('--mcts_max_depth', metavar='N', type=int,
                        default=100, help='the maximum depth of MCTS')
    parser.add_argument('--cpuct', metavar='N', type=int,
                        default=1,
                        help='an MCTS parameter c_puct')
    return parser


def make_an_episode(rank, state, nnets, action_space, args):
    device_num = rank % args.n_gpus
    nnet = nnets[device_num]
    device = torch.device(f'cuda:{device_num}')

    examples = []
    tmp_examples = []
    curr_scores = [0, 0]
    env = Env(action_space)

    if env.check_done(state) is not None:
        return examples

    mcts = MCTS(device, nnet, action_space, args)

    state = env.reset(state)
    while True:
        policy = mcts.get_policy(state)
        action = select_action(policy, state, action_space)

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


def simulate(batch, nnets, action_space, args):
    with mp.Pool(args.n_cpus) as pool:
        func = partial(make_an_episode,
                       nnets=nnets, action_space=action_space, args=args)
        unzipped = [{column: batch[column][i] for column in batch}  # unzip
                    for i in range(len(batch[list(batch.keys())[0]]))]
        examples_list = pool.starmap(func, enumerate(unzipped))
    return [example for examples in examples_list for example in examples]


def update_net(train_examples, nnet, optimizer, policy_loss_fn, value_loss_fn,
               iterations, batch_size):
    for _ in range(iterations):
        for batch in sequential_batch(train_examples, batch_size):
            states, policies, values = tuple(map(list, zip(*batch)))

            states = {k: torch.Tensor([dic[k] for dic in states]).cuda()
                      for k in states[0]}
            target_policies = torch.from_numpy(np.stack(policies)).cuda()
            target_values = map(lambda x: torch.Tensor(x).cuda(),
                                tuple(zip(*values)))

            policies, values = nnet(states)

            loss = policy_loss_fn(policies, target_policies) \
                + value_loss_fn(values[0], target_values[0]) \
                + value_loss_fn(values[1], target_values[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.save(nnet.state_dict(), 'models/temp.pt')
            print('Model saved')


def main():
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Training argument parser')
    parser = load_train_args(parser)
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    train_data = get_train_data()
    valid_data = get_valid_data()

    nnet = create_nnet(train_data, args)
    nnets = create_nnets(train_data, args, n_nnets=torch.cuda.device_count())
    optimizer = Adam(nnet.parameters(), lr=args.lr)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    action_space = ActionSpace()

    train_examples = deque(maxlen=args.examples_len)

    epoch = 0
    while True:
        for indice in random_batch(len(train_data), args.train_batch_size):
            epoch += 1
            print(f'Epoch {epoch}')

            copy_nnet(nnet, nnets)  # nnet -> nnets

            curr_examples = simulate(train_data[indice], nnets, action_space,
                                     args)
            train_examples.extend(curr_examples)

            update_net(train_examples, nnet, optimizer,
                       policy_loss_fn, value_loss_fn,
                       args.update_iterations, args.train_batch_size)

            test(valid_data, args)


if __name__ == '__main__':
    main()
