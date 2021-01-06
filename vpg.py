import argparse
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.models import *
from src.utils import *
from src.env import *


parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--dropout', type=float, default=0.5, metavar='F')
parser.add_argument('--l2', type=float, default=0.0, metavar='F')
parser.add_argument('--lr', type=float, default=1e-5, metavar='F')
parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
parser.add_argument('--batch-size', type=int, default=512, metavar='N')
parser.add_argument('--workers', type=int, default=4, metavar='N')
parser.add_argument('--cuda', type=int, default=0, metavar='N')
parser.add_argument('--simul', type=int, default=100, metavar='N')
# parser.add_argument('--epochs', type=int, default=10, metavar='N')
args = parser.parse_args()

torch.manual_seed(args.seed)

tag, device = init(args)
data = load_data()
num_bats, num_pits, num_teams = count_numbers(data)
_, _, tnewloader, _ = get_dataloaders(data, args)

print(f'# of train games: {len(tnewloader.dataset)}')

policy = Dynamics(num_bats, num_pits, num_teams, args.emb_dim, args.dropout).to(device)
policy.load_state_dict(torch.load(get_latest_file_path('models/dynamics')))
optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.l2)


def select_action(state):
    state = {key: value.to(device) for key, value in state.items()}
    bat_dest, run1_dest, run2_dest, run3_dest = policy(**state)
    bat_dest = Categorical(F.softmax(bat_dest, dim=1))
    run1_dest = Categorical(F.softmax(run1_dest, dim=1))
    run2_dest = Categorical(F.softmax(run2_dest, dim=1))
    run3_dest = Categorical(F.softmax(run3_dest, dim=1))
    bat_act = bat_dest.sample()
    run1_act = run1_dest.sample()
    run2_act = run2_dest.sample()
    run3_act = run3_dest.sample()
    policy.saved_log_probs.append(
        bat_dest.log_prob(bat_act) +
        run1_dest.log_prob(run1_act) +
        run2_dest.log_prob(run2_act) +
        run3_dest.log_prob(run3_act)
    )
    return bat_act.item(), run1_act.item(), run2_act.item(), run3_act.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    tb = SummaryWriter(f'runs/vpg_{tag}')

    running_reward = 0
    running_steps = 0
    i_episode = 0
    for features, targets in tnewloader:
        env = Env(features, targets)
        for _ in range(args.simul):
            i_episode += 1
            state, ep_reward = env.reset(), 0
            for t in range(1, 500):  # Don't infinite loop while learning
                actions = select_action(state)
                state, reward, done, _ = env.step(*actions)
                if args.render:
                    env.render()
                policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    running_steps = 0.05 * t + (1 - 0.05) * running_steps
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            finish_episode()
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tAverage steps: {}'.format(
                    i_episode, ep_reward, running_reward, running_steps))
                torch.save(policy.state_dict(), f'models/vpg/{tag}.pt')
                tb.add_scalar('running reward', running_reward, i_episode)

    tb.close()


if __name__ == '__main__':
    main()
