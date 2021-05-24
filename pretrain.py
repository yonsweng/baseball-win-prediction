import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from ActionSpace import ActionSpace
from test import load_test_args, test
from utils import get_train_dataset, get_valid_dataset, get_test_dataset, set_seeds, \
                  create_nnet, random_batch, to_input_batch


def load_pretrain_args(parser):
    parser.add_argument('--lr', metavar='F', type=float,
                        default=1e-6, help='the learning rate')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=512,
                        help='the number of plates to train for a batch')
    parser.add_argument('--update_epochs', metavar='N', type=int,
                        default=50,
                        help='the number of epochs to update the neural net')
    return parser


def get_target_policies(batch, action_space):
    actions = []
    for _, dests in enumerate(zip(batch['BAT_DEST_ID'], batch['RUN1_DEST_ID'],
                              batch['RUN2_DEST_ID'], batch['RUN3_DEST_ID'])):
        dests = tuple(min(4, dest) for dest in dests)
        action = action_space.to_action(dests)
        actions.append(action)
    return torch.tensor(actions, dtype=torch.long)


def get_target_values(batch):
    values = []
    for away_score_ct, home_score_ct, away_end_score_ct, home_end_score_ct \
        in zip(batch['AWAY_SCORE_CT'], batch['HOME_SCORE_CT'],
               batch['AWAY_END_SCORE_CT'], batch['HOME_END_SCORE_CT']):
        away_value = away_end_score_ct - away_score_ct
        home_value = home_end_score_ct - home_score_ct
        values.append([away_value, home_value])
    return torch.Tensor(values)


def main():
    parser = argparse.ArgumentParser(description='Pretraining argument parser')
    parser = load_pretrain_args(parser)
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    train_data = get_train_dataset()
    valid_data = get_valid_dataset()
    test_data = get_test_dataset()

    nnet = create_nnet(train_data, args)

    optimizer = Adam(nnet.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    action_space = ActionSpace()

    tb = SummaryWriter()

    best_score = 0

    for epoch in range(1, args.update_epochs + 1):
        print(f'Epoch {epoch}')

        for indice in random_batch(len(train_data), args.train_batch_size):
            batch = train_data[indice]
            input_batch = to_input_batch(batch, torch.device('cuda'))

            policies, values = nnet(input_batch)

            target_policies = get_target_policies(batch, action_space).cuda()
            target_values = get_target_values(batch).cuda()

            policy_loss = ce_loss(policies, target_policies)
            value_loss = mse_loss(values, target_values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = test(valid_data, nnet, args, tb, epoch)

        if accuracy > best_score:
            best_score = accuracy
            torch.save(nnet.module.state_dict(), 'models/pretrained.pt')

    nnet.module.load_state_dict(torch.load('models/pretrained.pt'))

    test(test_data, nnet, args, tb, args.update_epochs + 1)


if __name__ == '__main__':
    main()
