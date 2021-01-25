import os
import argparse
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from src.models import *
from src.utils import *
from src.env import *


def test(env, loader, model, cuda, args, low=0, high=0, verbose=0):
    print(f'Test started')
    model.eval()

    # get y_true
    y_true = []
    for ori_state, targets in loader:
        y_true += targets['result'][0].tolist()
    y_true = np.array(y_true)

    y_pred = np.zeros(len(loader.dataset))

    for i_simul in range(args.simul):
        local_y_pred = []

        for ori_state, targets in loader:
            steps = 0

            length = random.randint(low, high)
            state = env.reset(ori_state, targets)

            while True:
                steps += 1

                state = {key: value.to(cuda) for key, value in state.items()}
                bat_dest, run1_dest, run2_dest, run3_dest, pred = model(**state)
                pred = pred.squeeze()

                if steps >= length:
                    local_y_pred.append(pred.item())
                    break

                bat_act = Categorical(F.softmax(bat_dest.squeeze())).sample()
                run1_act = Categorical(F.softmax(run1_dest.squeeze())).sample()
                run2_act = Categorical(F.softmax(run2_dest.squeeze())).sample()
                run3_act = Categorical(F.softmax(run3_dest.squeeze())).sample()

                state, reward, done = env.step(bat_act, run1_act, run2_act, run3_act)

                if done:
                    local_y_pred.append(float((state['away_score_ct'] < state['home_score_ct']).item()))
                    break

        y_pred += np.array(local_y_pred)

    y_pred /= args.simul

    print(y_pred)
    print(y_true)

    threshold = 0.5
    precision = precision_score(y_true, y_pred > threshold)
    recall = recall_score(y_true, y_pred > threshold)
    accuracy = accuracy_score(y_true, y_pred > threshold)
    print(f'precision: {precision:.3f} \trecall: {recall:.3f} \taccuracy: {accuracy:.3f}')
    print(confusion_matrix(y_true, y_pred > threshold))

    return precision, recall, accuracy


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
    precision, recall, accuracy = test(env, testloader, model, device, args, args.min, args.max, 1)
