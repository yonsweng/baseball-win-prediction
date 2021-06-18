import argparse
import wandb
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from BaseballDataset import PlateDataset
from NNet import EventPredict
from utils import set_seeds, count_bats_pits_teams, count_events, accuracy, \
                  freeze, unfreeze


def load_train_args(parser):
    parser.add_argument('--lr', metavar='F', type=float,
                        default=1e-5,
                        help='the learning rate')
    parser.add_argument('--emb_lr', metavar='F', type=float,
                        default=1e-2,
                        help='the learning rate for embedding weights')
    parser.add_argument('--freeze_epochs', metavar='N', type=int,
                        default=15,
                        help='the epochs to freeze layers')
    parser.add_argument('--weight_decay', metavar='F', type=float,
                        default=1e-4,
                        help='weight decay')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=512,
                        help='the batch size for training')
    parser.add_argument('--test_batch_size', metavar='N', type=int,
                        default=512,
                        help='the batch size for test')
    parser.add_argument('--num_epochs', metavar='N', type=int,
                        default=100,
                        help='the number of epochs to train')
    parser.add_argument('--embedding_dim', metavar='N', type=int,
                        default=128,
                        help='the embedding dimension')
    parser.add_argument('--hidden_dim', metavar='N', type=int,
                        default=1024,
                        help='the hidden dimension')
    parser.add_argument('--num_linears', metavar='N', type=int,
                        default=3,
                        help='the number of linear layers')
    parser.add_argument('--seed', metavar='N', type=int, default=2021,
                        help='the random seed')
    return parser


def main():
    parser = argparse.ArgumentParser(description='Training argument parser')
    parser = load_train_args(parser)
    args = parser.parse_args()

    wandb.init(config=args)
    args = wandb.config

    set_seeds(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_train.csv',
        low_memory=False
    )
    valid_data = pd.read_csv(
        'input/mlbplaybyplay2010s_preprocessed/all2010_valid.csv',
        low_memory=False
    )

    num_batters, num_pitchers, num_teams = count_bats_pits_teams()
    events = count_events()
    event_to_index = {event: index for index, event in enumerate(events)}

    train_dataset = PlateDataset(train_data, event_to_index)
    valid_dataset = PlateDataset(valid_data, event_to_index)

    train_loader = DataLoader(train_dataset, args.train_batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.test_batch_size,
                              shuffle=False)

    model = EventPredict(
        num_batters,
        num_pitchers,
        num_teams,
        args.embedding_dim,
        len(events),
        args.hidden_dim,
        args.num_linears
    )
    model.to(device)
    wandb.watch(model)

    optimizer = Adam([
        {"params":
            list(model.bat_emb.parameters()) +
            list(model.pit_emb.parameters()) +
            list(model.team_emb.parameters()),
         "lr": args.emb_lr, "weight_decay": args.weight_decay},
        {"params":
            list(model.linear_in.parameters()) +
            list(model.linears.parameters()) +
            list(model.linear_out.parameters()),
         "lr": args.lr, "weight_decay": args.weight_decay}
    ])

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.emb_lr, args.lr],
        epochs=args.num_epochs,
        steps_per_epoch=(len(train_dataset)+args.train_batch_size-1
                         )//args.train_batch_size,

    )

    loss_fn = nn.CrossEntropyLoss()  # score

    freeze([model.linear_in, model.linears, model.linear_out])

    for epoch in range(1, 1 + args.num_epochs):
        model.train()
        sum_loss = 0

        for features, targets in train_loader:
            batch_size = features.shape[0]

            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)  # preds = (batch, 21)

            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_loss += loss.item() * batch_size

        metrics = {
            'train_loss': sum_loss / len(train_loader.dataset)
        }

        model.eval()
        sum_loss = 0
        best_accuracy = 0
        pred_list, target_list = [], []

        for features, targets in valid_loader:
            batch_size = features.shape[0]

            features = features.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                preds = model(features)  # preds = (batch, 21)

            loss = loss_fn(preds, targets)

            sum_loss += loss.item() * batch_size

            pred_list += preds.squeeze().tolist()
            target_list += targets.tolist()

        pred_list = torch.Tensor(pred_list)
        target_list = torch.Tensor(target_list)

        top_accuracy = accuracy(pred_list, target_list, topk=(3,))[0]

        metrics.update({
            'valid_loss': sum_loss / len(valid_loader.dataset),
            'accuracy': top_accuracy,
            'bat_emb': model.bat_emb.weight.detach().cpu().numpy(),
            'pit_emb': model.pit_emb.weight.detach().cpu().numpy(),
            'team_emb': model.team_emb.weight.detach().cpu().numpy(),
            'lr': scheduler.get_last_lr()[0],
        })

        wandb.log(metrics)

        # model save
        if top_accuracy > best_accuracy:
            best_accuracy = top_accuracy
            torch.save(model.state_dict(), 'models/embedding.pt')

        if epoch == args.freeze_epochs:
            unfreeze([model.linear_in, model.linears, model.linear_out])


if __name__ == '__main__':
    main()
