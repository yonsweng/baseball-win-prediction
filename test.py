import os
import argparse
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from src.models import *
from src.utils import *
from src.env import *


def test(env, loader, model, cuda, args, low=0, high=0, verbose=0):
    print(f'Test started')
    model.eval()
    N_SIMUL = args.simul  # should be odd
    i_game = 0
    true, false = 0, 0
    for ori_state, targets in loader:
        # if targets['result'][0].item() > 0.5:
        #     true += 1
        # else:
        #     false += 1
        local_true, local_false = 0, 0
        i_game += 1
        if verbose and i_game % 20 == 0:
            print(i_game)
        for _ in range(N_SIMUL):
            length = random.randint(low, high)
            steps = 0
            state = env.reset(ori_state, targets)
            while True:
                state = {key: value.to(cuda) for key, value in state.items()}
                if steps >= length:
                    _, _, _, _, pred = model(**state)
                    if abs(targets['result'][0].item() - pred.item()) <= 0.5:
                        local_true += 1
                    else:
                        local_false += 1
                    break
                steps += 1
                bat_act, run1_act, run2_act, run3_act = select_action(state, model)
                state, reward, done = env.step(bat_act, run1_act, run2_act, run3_act)
                if done:
                    if reward == 1:
                        local_true += 1
                    else:
                        local_false += 1
                    break
            # reset rewards and action buffer
            del model.saved_log_probs[:]
            del model.saved_rewards[:]
            del model.saved_preds[:]
        if local_true > local_false:
            true += 1
        else:
            false += 1
    accuracy = true / (true + false)
    print(f'accuracy: {accuracy}')
    return accuracy


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
    parser.add_argument('--seed', type=int, default=543, metavar='N')
    parser.add_argument('--workers', type=int, default=16, metavar='N')
    parser.add_argument('--cuda', type=int, default=1, metavar='N')
    parser.add_argument('--simul', type=int, default=1, metavar='N')  # should be odd
    parser.add_argument('--model', type=str, default='', metavar='S')
    parser.add_argument('--min', type=int, default=0, metavar='N')
    parser.add_argument('--max', type=int, default=0, metavar='N')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    file_path = args.model

    tag, device = init(args)
    data = load_data()
    num_bats, num_pits, num_teams = count_numbers(data)
    trainloader, validloader, tnewloader, vnewloader, testloader = get_dataloaders(data, args)

    print(f'# of plates: {len(trainloader.dataset)}')
    print(f'# of train games: {len(tnewloader.dataset)}')
    print(f'# of valid games: {len(vnewloader.dataset)}')
    print(f'# of test games: {len(testloader.dataset)}')

    env = Env()

    model = Model(num_bats, num_pits, num_teams, args.emb_dim, args.dropout, device).to(device)
    model.load_state_dict(torch.load(file_path))
    test(env, testloader, model, device, args, args.min, args.max, 1)
