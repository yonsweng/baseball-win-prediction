import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from src.dataset import *
from src.models import *
from src.preprocess import *
from src.utils import *
from src.env import *


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    CELoss = torch.nn.CrossEntropyLoss()

    lr_lambda = lambda x: x / args.warmup if x <= args.warmup else (x / args.warmup) ** -0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    clip = lambda x: x.where(x <= 4, torch.tensor([4], dtype=torch.long))

    best_loss = 99.9
    early_stopping = 0

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')

        # Training
        model.train()
        sum_loss = 0.
        for features, targets in trainloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest = model(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device)
            )
            loss = \
                CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device))
            sum_loss += bat_dest.shape[0] * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        tb.add_scalar('train loss', sum_loss / len(trainloader.dataset), epoch)
        tb.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        # Validation
        model.eval()
        sum_loss = 0.
        bat_dest_true, run1_dest_true, run2_dest_true, run3_dest_true = 0, 0, 0, 0
        for features, targets in validloader:
            fld_team_id = features['away_team_id'].where(
                features['bat_home_id'].type(torch.long) == 1,
                features['home_team_id']
            )
            bat_dest, \
            run1_dest, \
            run2_dest, \
            run3_dest = model(
                features['away_score_ct'].to(device),
                features['home_score_ct'].to(device),
                features['inn_ct'].to(device),
                features['bat_home_id'].to(device),
                features['outs_ct'].to(device),
                features['bat_id'].to(device),
                features['pit_id'].to(device),
                fld_team_id.to(device),
                features['base1_run_id'].to(device),
                features['base2_run_id'].to(device),
                features['base3_run_id'].to(device)
            )
            loss = \
                CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device))
            sum_loss += bat_dest.shape[0] * loss.item()
            # Calculate accuracies.
            bat_dest_true += (bat_dest.cpu().argmax(dim=1) == clip(targets['bat_dest']).squeeze()).sum().item()
            run1_dest_true += (run1_dest.cpu().argmax(dim=1) == clip(targets['run1_dest']).squeeze()).sum().item()
            run2_dest_true += (run2_dest.cpu().argmax(dim=1) == clip(targets['run2_dest']).squeeze()).sum().item()
            run3_dest_true += (run3_dest.cpu().argmax(dim=1) == clip(targets['run3_dest']).squeeze()).sum().item()
        tb.add_scalar('valid loss', sum_loss / len(validloader.dataset), epoch)
        tb.add_scalar('valid bat_dest acc.', bat_dest_true / len(validloader.dataset), epoch)
        tb.add_scalar('valid run1_dest acc.', run1_dest_true / len(validloader.dataset), epoch)
        tb.add_scalar('valid run2_dest acc.', run2_dest_true / len(validloader.dataset), epoch)
        tb.add_scalar('valid run3_dest acc.', run3_dest_true / len(validloader.dataset), epoch)

        # Save the best model.
        if sum_loss / len(validloader.dataset) < best_loss:
            best_loss = sum_loss / len(validloader.dataset)
            torch.save(model.state_dict(), f'./models/{PREFIX}/{tag}.pt')
            print('model saved.')
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt > args.patience:
                break


if __name__ == "__main__":
    PREFIX = 'dynamics'
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
    parser.add_argument('--cuda', type=int, default=0, metavar='N')
    args = parser.parse_args()

    tag, device = init(args)
    data = load_data()
    num_bats, num_pits, num_teams = count_numbers(data)
    trainloader, validloader, tnewloader, vnewloader = get_dataloaders(data, args)

    from test import test
    model = Dynamics(num_bats, num_pits, num_teams, args.emb_dim, args.dropout).to(device)
    tb = SummaryWriter(f'./runs/{PREFIX}_{tag}')
    train()
    tb.close()
    # model.load_state_dict(torch.load(get_latest_file_path('models/dynamics')))
    # test(vnewloader, model, device)
