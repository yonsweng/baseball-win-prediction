import os
import argparse
import wandb
import torch.multiprocessing as multiprocessing
from multiprocessing.pool import Pool
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from functools import partial
from BaseballDataset import collate_fn
from NNet import EventPredict
from test import load_test_args
from utils import count_bats_pits_teams, count_events, set_seeds, \
                  get_train_dataset, get_valid_dataset, create_nnets


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


def load_train_args(parser):
    parser.add_argument('--lr', metavar='F', type=float,
                        default=1e-4,
                        help='the learning rate')
    parser.add_argument('--weight_decay', metavar='F', type=float,
                        default=1e-4,
                        help='weight decay')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=256,
                        help='the batch size for training')
    parser.add_argument('--num_epochs', metavar='N', type=int,
                        default=200,
                        help='the number of epochs to train')
    parser.add_argument('--embedding_size', metavar='N', type=int,
                        default=128,
                        help='the embedding size')
    parser.add_argument('--hidden_size', metavar='N', type=int,
                        default=256,
                        help='the hidden size')
    parser.add_argument('--num_blocks', metavar='N', type=int,
                        default=0,
                        help='the number of residual blocks of the nnets')
    parser.add_argument('--num_linears', metavar='N', type=int,
                        default=7,
                        help='the number of linear blocks of the nnets')
    parser.add_argument('--load_emb', action='store_true',
                        help='load the embeddings')
    return parser


def train(fold, args, device) -> float:
    run = wandb.init(group='train', config=args)
    with run:
        args = wandb.config

        train_dataset = get_train_dataset(fold)
        valid_dataset = get_valid_dataset(fold)

        train_loader = DataLoader(train_dataset, args.train_batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, args.test_batch_size,
                                  shuffle=False, collate_fn=collate_fn)

        num_batters, num_pitchers, num_teams = count_bats_pits_teams()
        events = count_events()

        predict = create_nnets(train_dataset, args)
        predict.to(device)

        if args.load_emb:
            os.system(f'python embedding.py --fold {fold}')

            embedding_model = EventPredict(num_batters, num_pitchers,
                                           num_teams, args.embedding_size,
                                           len(events), 1024, 3)
            embedding_model.to(device)
            embedding_model.load_state_dict(
                torch.load(f'models/embedding_{fold}.pt'))

            predict.bat_emb.load_state_dict(
                embedding_model.bat_emb.state_dict())
            predict.pit_emb.load_state_dict(
                embedding_model.pit_emb.state_dict())
            predict.team_emb.load_state_dict(
                embedding_model.team_emb.state_dict())

        wandb.watch(predict)

        optimizer = Adam(
            predict.parameters(),
            args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr,
            epochs=args.num_epochs,
            steps_per_epoch=(len(train_dataset)+args.train_batch_size-1
                             )//args.train_batch_size
        )

        mse_loss = nn.MSELoss()  # score

        best_accuracy = 0

        for _ in range(1, 1 + args.num_epochs):
            predict.train()
            sum_loss = 0

            for features, _, score_targets, _, inns in train_loader:
                batch_size = score_targets.shape[0]

                score_targets = score_targets / inns * 9

                first_features = {}
                for key in features:
                    first_features[key] = features[key][0]

                for key in first_features:
                    first_features[key] = first_features[key].to(device)
                score_targets = score_targets.to(device)

                score_preds = predict(first_features)  # (batch, 2)

                loss = mse_loss(score_preds, score_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                sum_loss += loss.item() * batch_size

            metrics = {
                'train_loss': sum_loss / len(train_loader.dataset)
            }

            predict.eval()
            sum_loss = 0
            num_trues = 0

            with torch.no_grad():
                for features, _, score_targets, _, _ in valid_loader:
                    batch_size = score_targets.shape[0]

                    first_features = {}
                    for key in features:
                        first_features[key] = features[key][0]

                    for key in first_features:
                        first_features[key] = first_features[key].to(device)
                    score_targets = score_targets.to(device)

                    score_preds = predict(first_features)  # (batch, 2)

                    loss = mse_loss(score_preds, score_targets)

                    sum_loss += loss.item() * batch_size

                    num_trues += \
                        ((score_preds[:, 0] < score_preds[:, 1]) ==
                         (score_targets[:, 0] < score_targets[:, 1])).sum()

            accuracy = num_trues / len(valid_loader.dataset)

            metrics.update({
                'valid_loss': sum_loss / len(valid_loader.dataset),
                'accuracy': accuracy,
                'bat_emb': predict.bat_emb.weight.detach().cpu().numpy(),
                'pit_emb': predict.pit_emb.weight.detach().cpu().numpy(),
                'team_emb': predict.team_emb.weight.detach().cpu().numpy(),
                'lr': scheduler.get_last_lr()[0]
            })

            # best model save
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if args.load_emb:
                    torch.save(predict.state_dict(),
                               f'models/with_emb_{fold}.pt')
                else:
                    torch.save(predict.state_dict(),
                               f'models/without_emb_{fold}.pt')

            wandb.log(metrics)

    return best_accuracy


def main():
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Training argument parser')
    parser = load_train_args(parser)
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pool = NoDaemonPool(5)
    train_with_args = partial(train, args=args, device=device)
    cv_result = pool.map(train_with_args, range(5))
    pool.close()
    pool.join()

    run = wandb.init(group='train', config=args)
    with run:
        wandb.log({"cv_accuracy": list(cv_result)})


if __name__ == '__main__':
    main()
