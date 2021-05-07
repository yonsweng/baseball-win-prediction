import argparse
from collections import deque
from functools import partial
from itertools import cycle
from tqdm import tqdm
import multiprocessing.pool as mpp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from istarmap import istarmap
from Env import Env
from ActionSpace import ActionSpace
from MCTS import MCTS
from test import load_test_args, test
from utils import select_action, set_seeds, copy_nnet, to_input_batch, \
                  create_nnet, create_nnets, get_train_data, get_valid_data, \
                  random_batch, sequential_list_batch


def load_train_args(parser):
    parser.add_argument('--lr', metavar='F', type=float,
                        default=1e-6, help='the learning rate')
    parser.add_argument('--examples_len', metavar='N', type=int,
                        default=100000,
                        help='the maximum length of train examples')
    parser.add_argument('--n_cpus', metavar='N', type=int,
                        default=16,
                        help='the number of processors to use')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=16*20,
                        help='the number of episodes to simulate for an epoch')
    parser.add_argument('--update_epochs', metavar='N', type=int,
                        default=10,
                        help='the number of epochs to update the neural net')

    # MCTS arguments
    parser.add_argument('--num_mcts_sims', metavar='N', type=int,
                        default=500, help='the number of MCTS simulations')
    parser.add_argument('--mcts_max_depth', metavar='N', type=int,
                        default=5, help='the maximum depth of MCTS')
    parser.add_argument('--cpuct', metavar='N', type=int,
                        default=1,
                        help='an MCTS parameter c_puct')
    return parser


def make_an_episode(env, mcts, action_space, nnet, state):
    examples = []
    tmp_examples = []
    curr_scores = [0, 0]

    if env.check_done(state) is not None:
        return examples

    state = env.reset(state)
    for _ in range(100):  # max 100 steps
        policy = mcts.get_policy(state, nnet)
        action = select_action(policy, state, action_space)

        prev_state = state.copy()

        state, runs_scored, done, _ = env.step(action)

        curr_scores[0] += runs_scored[0]
        curr_scores[1] += runs_scored[1]

        tmp_examples.append((prev_state, policy, tuple(curr_scores)))

        if done:
            break

    for example in tmp_examples:
        future_runs = (curr_scores[0] - example[2][0],
                       curr_scores[1] - example[2][1])
        examples.append((example[0], example[1], future_runs))

    return examples


def simulate(batch, nnets, action_space, args):
    with mp.Pool(args.n_cpus) as pool:
        env = Env(action_space)
        mcts = MCTS(action_space, args)
        func = partial(make_an_episode, env, mcts, action_space)
        states = [{column: batch[column][i] for column in batch}
                  for i in range(len(batch[list(batch.keys())[0]]))]  # unzip
        examples_list = list(tqdm(
            pool.istarmap(func, zip(cycle(nnets), states)),
            total=len(states)
        ))
    return [example for examples in examples_list for example in examples]


def update_net(train_examples, nnet, optimizer, policy_loss_fn, value_loss_fn,
               args, tb, epoch):
    for _ in tqdm(range(args.update_epochs)):
        epoch_policy_loss, epoch_value_loss = 0, 0

        for batch in sequential_list_batch(list(train_examples),
                                           args.train_batch_size):
            states, policies, values = tuple(map(list, zip(*batch)))

            states = {k: np.array([dic[k] for dic in states])
                      for k in states[0]}
            input_batch = to_input_batch(states, torch.device('cuda'))

            target_policies = torch.from_numpy(np.stack(policies)).cuda()

            target_values = tuple(map(lambda x: torch.Tensor(x).cuda(),
                                      zip(*values)))

            policies, values = nnet(input_batch)

            policies = F.log_softmax(policies, dim=1)

            policy_loss = policy_loss_fn(policies, target_policies)
            value_loss = value_loss_fn(values[:, 0], target_values[0]) + \
                value_loss_fn(values[:, 1], target_values[1])
            loss = policy_loss + value_loss

            epoch_policy_loss += policy_loss.item() * len(policies)
            epoch_value_loss += value_loss.item() * len(policies)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_policy_loss /= len(train_examples)
        epoch_value_loss /= len(train_examples)

    tb.add_scalar('train policy loss', epoch_policy_loss, epoch)
    tb.add_scalar('train value loss', epoch_value_loss, epoch)

    torch.save(nnet.module.state_dict(), 'models/temp.pt')
    print('Model saved')


def main():
    mp.set_start_method('spawn')
    mpp.Pool.istarmap = istarmap  # for tqdm

    parser = argparse.ArgumentParser(description='Training argument parser')
    parser = load_train_args(parser)
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    train_data = get_train_data()
    valid_data = get_valid_data()

    nnet = create_nnet(train_data, args)
    nnet.module.load_state_dict(torch.load(f'models/{args.load}'))
    nnets = create_nnets(train_data, args, n_nnets=torch.cuda.device_count())

    optimizer = Adam(nnet.parameters(), lr=args.lr)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()

    action_space = ActionSpace()

    train_examples = deque(maxlen=args.examples_len)

    tb = SummaryWriter()  # tensorboard writer

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
                       policy_loss_fn, value_loss_fn, args, tb, epoch)

            test(valid_data, nnet, args, tb, epoch)


if __name__ == '__main__':
    main()
