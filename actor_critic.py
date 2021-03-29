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


def batch_train(transitions):
    s_lst, a_lst, r_lst, s_prime_lst, done_lst, y_lst = [], [], [], [], [], []
    for s, a, r, s_prime, done, y in transitions:
        s_lst.append(s)
        a_lst.append(a)
        r_lst.append([r])
        s_prime_lst.append(s_prime)
        done_lst.append([done])
        y_lst.append([y])

    # make s_batch
    s_batch = {key: [] for key in s_lst[0]}
    for s in s_lst:
        for key, value in s.items():
            s_batch[key].append(value)
    for key in s_batch:
        s_batch[key] = torch.cat(s_batch[key]).to(device)

    a_batch = torch.tensor(a_lst, device=device, dtype=torch.long).transpose(1, 0).reshape(4, -1, 1)
    r_batch = torch.tensor(r_lst, device=device, dtype=torch.float)
    y_batch = torch.tensor(y_lst, device=device, dtype=torch.float)

    # make s_prime_batch
    s_prime_batch = {key: [] for key in s_prime_lst[0]}
    for s_prime in s_prime_lst:
        for key, value in s_prime.items():
            s_prime_batch[key].append(value)
    for key in s_prime_batch:
        s_prime_batch[key] = torch.cat(s_prime_batch[key]).to(device)

    done_batch = torch.tensor(done_lst, device=device, dtype=torch.float)

    # get td-target and advantage
    v_s = (1 - 2 * y_batch) * (1 - 2 * model.v(**s_batch))
    v_s_prime = (1 - 2 * y_batch) * (1 - 2 * model.v(**s_prime_batch))
    td_target = r_batch + args.gamma * v_s_prime * (1 - done_batch)
    advantage = td_target - v_s

    policy = model(**s_batch)
    policy_a = torch.stack(policy).gather(2, a_batch)

    loss = -torch.log(policy_a) * advantage.detach() + F.smooth_l1_loss(v_s, td_target.detach())

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


def train():
    tb = SummaryWriter(f'./runs/{tag}')

    transitions = deque(maxlen=args.prop_interval)
    saved_steps = deque(maxlen=args.log_interval)
    bat_out_probs = deque(maxlen=1000)
    best_accuracy = 0

    env = Env()

    # test before RL
    precision, recall, accuracy = test(env, vnewloader, model, device, args, 0, 0)
    tb.add_scalar('accuracy', accuracy, 0)

    # run inifinitely many episodes
    i_episode = 0
    steps = 0
    for epoch in range(args.epochs):
        print('Epoch', epoch)

        for state, targets in trainloader:
            model.train()

            i_episode += 1
            done = False

            y = targets['result'][0].item()
            state = env.reset(state, targets)

            for t in range(args.epi_len):
                steps += 1

                state = {key: value.to(device) for key, value in state.items()}
                bat_dest, run1_dest, run2_dest, run3_dest = model(**state)

                # select action from policy
                actions = select_action(bat_dest, run1_dest, run2_dest, run3_dest)

                # take the action
                next_state, reward, done = env.step(*actions)

                # TD(0)
                transition = (state, actions, reward, next_state, done, y)
                transitions.append(transition)

                # for tensorboard
                bat_out_probs.append(bat_dest[:, 0].item())

                state = next_state

                # perform backprop
                if steps % args.prop_interval == 0:
                    batch_train(transitions)

                if done:
                    break

            # log results
            if i_episode % args.log_interval == 0:
                print('Episode {}'.format(i_episode))
                tb.add_histogram('bat_emb', model.bat_emb.weight, i_episode)
                tb.add_histogram('pit_emb', model.pit_emb.weight, i_episode)
                tb.add_histogram('team_emb', model.team_emb.weight, i_episode)
                tb.add_histogram('bat_out_probs', np.array(bat_out_probs), i_episode)

            # validation
            if i_episode % args.test_interval == 0:
                precision, recall, accuracy = test(env, vnewloader, model, device, args, 0, 0)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'models/{tag}.pt')
                    print('Model saved')

                tb.add_scalar('precision', precision, i_episode)
                tb.add_scalar('recall', recall, i_episode)
                tb.add_scalar('accuracy', accuracy, i_episode)
                tb.flush()

    tb.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G')
    parser.add_argument('--epi-len', type=int, default=5, metavar='N')
    parser.add_argument('--prop-interval', type=int, default=50, metavar='N')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N')
    parser.add_argument('--test-interval', type=int, default=2000, metavar='N')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='F')
    parser.add_argument('--l2', type=float, default=1e-3, metavar='F')
    parser.add_argument('--lr', type=float, default=2e-6, metavar='F')
    parser.add_argument('--value-ratio', type=float, default=1.0, metavar='F')
    parser.add_argument('--emb-dim', type=int, default=8, metavar='N')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--workers', type=int, default=19, metavar='N')
    parser.add_argument('--cuda', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--simul', type=int, default=1, metavar='N')
    parser.add_argument('--seed', type=int, default=543, metavar='N')
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
    # model.load_state_dict(torch.load('models/pretrain/01-28-08-41-08_dropout=0.3_l2=0.001_lr=2e-06_result_ratio=1_emb_dim=128_batch_size=256_epochs=100_patience=5_seed=543_workers=32_cuda=1.pt'))
    model.load_state_dict(torch.load(get_latest_file_path('models/pretrain')))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    eps = np.finfo(np.float32).eps.item()

    train()
