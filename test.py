import os
import argparse
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from src.models import *
from src.utils import *
from src.env import *


def test(loader, model, cuda, args, low=0, high=20):
    print(f'Test started')
    random.seed(args.seed)
    model.eval()
    N_SIMUL = args.simul  # should be odd
    true, false = 0, 0
    for policy_state, value_state, _, value_target in loader:
        local_true, local_false = 0, 0
        env = Env(policy_state, value_state)
        for _ in range(N_SIMUL):
            length = random.randint(low, high)
            steps = 0
            policy_state, value_state = env.reset()
            while True:
                if steps >= length:
                    total_state = {**policy_state, **value_state}
                    total_state = {key: value.to(cuda) for key, value in total_state.items()}
                    _, _, _, _, value = model(**total_state)
                    if abs(value_target['value'][0].item() - value.item()) <= 0.5:
                        local_true += 1
                    else:
                        local_false += 1
                    break
                steps += 1
                total_state = {**policy_state, **value_state}
                total_state = {key: value.to(cuda) for key, value in total_state.items()}
                bat_act, run1_act, run2_act, run3_act = select_action(total_state, model)
                policy_state, value_state, result, done = env.step(bat_act, run1_act, run2_act, run3_act)
                if done:
                    if value_target['value'][0].item() == result:
                        local_true += 1
                    else:
                        local_false += 1
                    break
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
    args = parser.parse_args()

    file_path = args.model

    tag, device = init(args)
    data = load_data()
    num_bats, num_pits, num_teams = count_numbers(data)
    trainloader, validloader, tnewloader, vnewloader, testloader = get_dataloaders(data, args)

    print(f'# of plates: {len(trainloader.dataset)}')
    print(f'# of train games: {len(tnewloader.dataset)}')
    print(f'# of valid games: {len(vnewloader.dataset)}')
    print(f'# of test games: {len(testloader.dataset)}')

    model = Model(num_bats, num_pits, num_teams, args.emb_dim, args.dropout, device).to(device)
    model.load_state_dict(torch.load(file_path))
    test(testloader, model, device, args)
