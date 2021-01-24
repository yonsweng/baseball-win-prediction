import argparse
import gym
import numpy as np
import random
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
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--prop-interval', type=int, default=10, metavar='N',
                    help='interval between backpropagation (default: 10)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='interval between training status logs (default: 20)')
parser.add_argument('--test-interval', type=int, default=200, metavar='N',
                    help='interval between tests (default: 200)')
parser.add_argument('--dropout', type=float, default=0.5, metavar='F')
parser.add_argument('--l2', type=float, default=1e-4, metavar='F')
parser.add_argument('--lr', type=float, default=3e-6, metavar='F')
parser.add_argument('--value-ratio', type=float, default=1.0, metavar='F')
parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
parser.add_argument('--batch-size', type=int, default=1, metavar='N')
parser.add_argument('--workers', type=int, default=12, metavar='N')
parser.add_argument('--cuda', type=int, default=1, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--simul', type=int, default=1, metavar='N')
parser.add_argument('--epi-len', type=int, default=20, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

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


def finish_episode(y, tb, i_episode):
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss

    curr_preds = model.saved_preds[:-1]
    next_preds = model.saved_preds[1:]

    for log_prob, curr_pred, next_pred, reward in zip(model.saved_log_probs, curr_preds, next_preds, model.saved_rewards):
        curr_value = (-1) ** y * (1 - 2 * curr_pred)
        next_value = (-1) ** y * (1 - 2 * next_pred)

        advantage = reward + args.gamma * next_value.detach() - curr_value

        # calculate actor (policy) loss
        policy_loss = -log_prob * advantage.detach()
        policy_loss = torch.clamp(policy_loss, min=-2., max=2.)
        policy_losses.append(policy_loss)

        # calculate critic (value) loss
        # value_losses.append(advantage ** 2)
        value_losses.append(F.smooth_l1_loss(curr_value, reward + args.gamma * next_value.detach()))

    # sum up all the values of policy_losses and value_losses
    policy_losses = torch.stack(policy_losses)
    value_losses = torch.stack(value_losses)
    loss = policy_losses.mean() + args.value_ratio * value_losses.mean()

    # TensorBoard
    tb.add_histogram('policy loss', policy_losses, i_episode)
    tb.add_histogram('value loss', value_losses, i_episode)

    # perform backprop
    if i_episode % args.prop_interval == 0:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        loss.backward()

    # reset rewards and action buffer
    del model.saved_log_probs[:]
    del model.saved_rewards[:]
    del model.saved_preds[:]


def main():
    tb = SummaryWriter(f'./runs/{tag}')

    saved_steps = deque(maxlen=args.log_interval)
    preds_list = deque(maxlen=1000)
    best_accuracy = 0

    env = Env()

    # Test before RL
    accuracy = test(env, vnewloader, model, device, args, 0, 0)
    tb.add_scalar('accuracy', accuracy, 0)

    # run inifinitely many episodes
    i_episode = 0
    for epoch in range(args.epochs):
        print('Epoch', epoch)
        for state, targets in trainloader:
            model.train()
            i_episode += 1

            state = env.reset(state, targets)

            steps = 0
            done = False

            for t in range(args.epi_len):
                steps += 1

                # select action from policy
                state = {key: value.to(device) for key, value in state.items()}
                actions = select_action(state, model)

                # take the action
                state, reward, done = env.step(*actions)

                # save the reward
                model.saved_rewards.append(reward)  # reward: float

                if done:
                    break

            # one more step futher
            if done:
                model.saved_preds.append(torch.tensor([0.5], device=device))
            else:
                state = {key: value.to(device) for key, value in state.items()}
                _, _, _, _, pred = model(**state)
                model.saved_preds.append(pred[0])

            saved_steps.append(steps)

            # Save preds for TensorBoard
            for pred in model.saved_preds:
                preds_list.append(pred.item())

            # perform backprop
            finish_episode(targets['result'][0].item(), tb, i_episode)

            # log results
            if i_episode % args.log_interval == 0:
                print('Episode {}\tlen {:.2f}'.format(i_episode, sum(saved_steps) / len(saved_steps)))
                tb.add_histogram('pred', np.array(preds_list), i_episode)

            # Validation
            if i_episode % args.test_interval == 0:
                accuracy = test(env, vnewloader, model, device, args, 0, 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'models/{tag}.pt')
                    print('Model saved')
                tb.add_scalar('accuracy', accuracy, i_episode)
                tb.flush()

    tb.close()


if __name__ == '__main__':
    main()
