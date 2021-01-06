import os
import argparse
import torch.nn.functional as F
from torch.distributions import Categorical
from src.models import *
from src.utils import *
from src.env import *


def test(vnewloader, model, device):
    print('test start')
    model.eval()
    N_SIMUL = args.simul  # should be odd
    episodes = 0
    tot_steps = 0
    true, false = 0, 0
    for features, targets in vnewloader:
        local_true, local_false = 0, 0
        env = Env(features, targets)
        for _ in range(N_SIMUL):
            steps = 0
            episodes += 1
            state = env.reset()
            while True:
                steps += 1
                state = {key: value.to(device) for key, value in state.items()}
                bat_dest, run1_dest, run2_dest, run3_dest = model(**state)
                bat_dest = Categorical(F.softmax(bat_dest, dim=1))
                run1_dest = Categorical(F.softmax(run1_dest, dim=1))
                run2_dest = Categorical(F.softmax(run2_dest, dim=1))
                run3_dest = Categorical(F.softmax(run3_dest, dim=1))
                bat_act = bat_dest.sample().item()
                run1_act = run1_dest.sample().item()
                run2_act = run2_dest.sample().item()
                run3_act = run3_dest.sample().item()
                state, reward, done, _ = env.step(bat_act, run1_act, run2_act, run3_act)
                # print(state, reward, done, steps)
                if done:
                    if reward == 1:
                        local_true += 1
                    elif reward == -1:
                        local_false += 1
                    tot_steps += steps
                    break
        if local_true > local_false:
            true += 1
        else:
            false += 1
    print(f'accuracy: {true / (true + false)}')
    print(f'avg. steps: {tot_steps / episodes}')
    print('test end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 자주 바뀌는 순.
    parser.add_argument('--dropout', type=float, default=0.2, metavar='F')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='F')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='F')
    parser.add_argument('--warmup', type=int, default=1000, metavar='N')
    parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N')
    parser.add_argument('--epochs', type=int, default=50, metavar='N')
    parser.add_argument('--patience', type=int, default=3, metavar='N')
    parser.add_argument('--seed', type=int, default=777, metavar='N')
    parser.add_argument('--workers', type=int, default=16, metavar='N')
    parser.add_argument('--cuda', type=int, default=1, metavar='N')
    parser.add_argument('--simul', type=int, default=1, metavar='N')
    args = parser.parse_args()

    file_path = input('policy file: ')

    tag, device = init(args)
    data = load_data()
    num_bats, num_pits, num_teams = count_numbers(data)
    trainloader, validloader, tnewloader, vnewloader = get_dataloaders(data, args)

    model = Dynamics(num_bats, num_pits, num_teams, args.emb_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(file_path))
    test(vnewloader, model, device)
