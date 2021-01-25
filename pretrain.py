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
    BCELoss = torch.nn.BCELoss()

    clip = lambda x: x.where(x <= 4, torch.tensor([4]))

    best_loss = 99.9
    early_stopping = 0

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}')

        # Training
        model.train()
        epoch_loss = 0

        for state, targets in trainloader:
            state = {key: value.to(device) for key, value in state.items()}
            bat_dest, run1_dest, run2_dest, run3_dest, pred = model(**state)

            loss = CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) \
                 + CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) \
                 + CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) \
                 + CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device)) \
                 + args.result_ratio * BCELoss(pred, targets['result'].to(device))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * pred.shape[0]

        epoch_loss /= len(trainloader.dataset)
        tb.add_scalar('train loss', epoch_loss, epoch)

        # Validation
        model.eval()
        epoch_loss = 0
        preds = []

        for state, targets in validloader:
            state = {key: value.to(device) for key, value in state.items()}
            bat_dest, run1_dest, run2_dest, run3_dest, pred = model(**state)

            loss = CELoss(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) \
                 + CELoss(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) \
                 + CELoss(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) \
                 + CELoss(run3_dest, clip(targets['run3_dest']).squeeze().to(device)) \
                 + args.result_ratio * BCELoss(pred, targets['result'].to(device))

            epoch_loss += loss.item() * pred.shape[0]

            preds += pred.tolist()

        epoch_loss /= len(validloader.dataset)
        tb.add_scalar('valid loss', epoch_loss, epoch)
        # tb.add_histogram('pred', np.array(preds), epoch)

        # draw histogram of weights on the tensorboard
        tb.add_histogram('bat_emb', model.bat_emb.weight, epoch)
        tb.add_histogram('pit_emb', model.pit_emb.weight, epoch)
        tb.add_histogram('team_emb', model.team_emb.weight, epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'models/pretrain/{tag}.pt')
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping > args.patience:
                break

        tb.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 자주 바뀌는 순.
    parser.add_argument('--dropout', type=float, default=0.5, metavar='F')
    parser.add_argument('--l2', type=float, default=1e-2, metavar='F')
    parser.add_argument('--lr', type=float, default=2e-6, metavar='F')
    parser.add_argument('--result-ratio', type=float, default=1, metavar='F')
    parser.add_argument('--emb-dim', type=int, default=32, metavar='N')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N')
    parser.add_argument('--epochs', type=int, default=50, metavar='N')
    parser.add_argument('--patience', type=int, default=3, metavar='N')
    parser.add_argument('--seed', type=int, default=543, metavar='N')
    parser.add_argument('--workers', type=int, default=4, metavar='N')
    parser.add_argument('--cuda', type=int, default=1, metavar='N')
    args = parser.parse_args()

    tag, device = init(args)
    data = load_data()
    num_bats, num_pits, num_teams = count_numbers(data)
    trainloader, validloader, tnewloader, vnewloader, testloader = get_dataloaders(data, args)

    model = Model(num_bats, num_pits, num_teams, args.emb_dim, args.dropout, device).to(device)

    print(tag)
    tb = SummaryWriter(f'./runs/{tag}')
    train()
    tb.close()
