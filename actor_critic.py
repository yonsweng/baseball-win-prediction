import argparse
import gym
import numpy as np
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.dataset import *
from src.models import *
from src.preprocess import *
from src.utils import *
from src.env import *
from test import test


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 20)')
parser.add_argument('--test-interval', type=int, default=200, metavar='N',
                    help='interval between tests (default: 200)')
parser.add_argument('--dropout', type=float, default=0.5, metavar='F')
parser.add_argument('--l2', type=float, default=0.0, metavar='F')
parser.add_argument('--lr', type=float, default=1e-6, metavar='F')
parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
parser.add_argument('--batch-size', type=int, default=1, metavar='N')
parser.add_argument('--workers', type=int, default=16, metavar='N')
parser.add_argument('--cuda', type=int, default=1, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--simul', type=int, default=1, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

tag, device = init(args)
data = load_data()
num_bats, num_pits, num_teams = count_numbers(data)
trainloader, validloader, tnewloader, vnewloader, testloader = get_dataloaders(data, args)

print(f'# of plates: {len(trainloader.dataset)}')
print(f'# of train games: {len(tnewloader.dataset)}')
print(f'# of valid games: {len(vnewloader.dataset)}')
print(f'# of test games: {len(testloader.dataset)}')

model = Model(num_bats, num_pits, num_teams, args.emb_dim, args.dropout, device).to(device)
model.load_state_dict(torch.load(get_latest_file_path('models/pretrain')))
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

eps = np.finfo(np.float32).eps.item()


def finish_episode(y):
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss

    curr_values = model.values[:-1]
    next_values = model.values[1:]

    for log_prob, curr_value, next_value in zip(model.saved_log_probs, curr_values, next_values):
        # calculate actor (policy) loss
        policy_losses.append(-log_prob * (next_value.detach() - curr_value.detach()) * (y - curr_value.detach())
            / (eps + curr_value.detach() * (1 - curr_value.detach())))

        # calculate critic (value) loss using BCE loss
        value_losses.append(F.binary_cross_entropy(curr_value, next_value.detach()))

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()

    # reset rewards and action buffer
    del model.values[:]
    del model.saved_log_probs[:]


def main():
    tb = SummaryWriter(f'./runs/{tag}')

    iterator = iter(trainloader)
    saved_steps = deque(maxlen=args.log_interval)
    best_accuracy = 0

    # Test before RL
    accuracy = test(vnewloader, model, device, args)
    tb.add_scalar('accuracy', accuracy, 0)

    # run inifinitely many episodes
    for i_episode in count(1):
        model.train()

        policy_state, value_state, _, value_target = next(iterator)

        env = Env(policy_state, value_state)
        policy_state, value_state = env.reset()

        steps = 0
        done = False

        for t in range(1, 100):
            steps += 1

            # select action from policy
            state = {**policy_state, **value_state}
            state = {key: value.to(device) for key, value in state.items()}
            actions = select_action(state, model)

            # take the action
            policy_state, value_state, result, done = env.step(*actions)

            if done:
                model.values.append(torch.tensor([result], dtype=torch.float, device=device))
                break

        if not done:
            total_state = {**policy_state, **value_state}
            total_state = {key: value.to(device) for key, value in total_state.items()}
            _, _, _, _, value = model(**total_state)
            model.values.append(value.reshape(-1))

        saved_steps.append(steps)

        # perform backprop
        finish_episode(value_target['value'][0].item())

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tlen {:.2f}'.format(i_episode, sum(saved_steps) / len(saved_steps)))
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        if i_episode % args.test_interval == 0:
            accuracy = test(vnewloader, model, device, args)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'models/{tag}.pt')
                print('Model saved')
            tb.add_scalar('accuracy', accuracy, i_episode)

    tb.close()


if __name__ == '__main__':
    main()
