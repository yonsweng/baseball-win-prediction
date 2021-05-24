import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, \
                               pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from BaseballDataset import collate_fn
from test import load_test_args
from utils import set_seeds, get_train_dataset, get_valid_dataset, create_nnets


def load_train_args(parser):
    parser.add_argument('--lr', metavar='F', type=float,
                        default=1e-5,
                        help='the learning rate')
    parser.add_argument('--train_batch_size', metavar='N', type=int,
                        default=32,
                        help='the batch size for training')
    parser.add_argument('--num_epochs', metavar='N', type=int,
                        default=100,
                        help='the number of epochs to train')
    parser.add_argument('--represent_size', metavar='N', type=int,
                        default=256,
                        help='the representation size')
    parser.add_argument('--embedding_size', metavar='N', type=int,
                        default=256,
                        help='the embedding size')
    parser.add_argument('--hidden_size', metavar='N', type=int,
                        default=256,
                        help='the hidden size')
    parser.add_argument('--num_blocks', metavar='N', type=int,
                        default=5,
                        help='the number of residual blocks of the nnets')
    parser.add_argument('--max_seq_len', metavar='N', type=int,
                        default=100,
                        help='the maximum number of PAs')
    parser.add_argument('--state_loss_coef', metavar='F', type=float,
                        default=10,
                        help='the coefficient of state loss')
    return parser


def apply_mask(tensor, lengths, device):
    mask = pad_sequence(
        [torch.ones(length) for length in lengths]
    ).unsqueeze(-1).repeat(
        1, 1, tensor.shape[-1]
    ).to(device)
    return tensor * mask


def main():
    parser = argparse.ArgumentParser(description='Training argument parser')
    parser = load_train_args(parser)
    parser = load_test_args(parser)
    args = parser.parse_args()

    set_seeds(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataset = get_train_dataset()
    valid_dataset = get_valid_dataset()

    train_loader = DataLoader(train_dataset, args.train_batch_size,
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, args.test_batch_size,
                              shuffle=False, collate_fn=collate_fn)

    represent, is_done, predict = create_nnets(train_dataset, args)
    rnn = nn.GRU(
        input_size=args.represent_size,
        hidden_size=args.represent_size,
        num_layers=1
    )

    represent.to(device)
    is_done.to(device)
    predict.to(device)
    rnn.to(device)

    optimizer = Adam(
        list(represent.parameters()) +
        list(is_done.parameters()) +
        list(predict.parameters()) +
        list(rnn.parameters()),
        args.lr
    )

    cos_loss = nn.CosineEmbeddingLoss()  # state
    bce_loss = nn.BCELoss()  # done
    mse_loss = nn.MSELoss()  # score

    tb = SummaryWriter()  # tensorboard writer

    for epoch in range(1, 1 + args.num_epochs):
        represent.train()
        is_done.train()
        predict.train()
        rnn.train()
        sum_state_loss, sum_done_loss, sum_score_loss, sum_loss = 0, 0, 0, 0

        for features, done_targets, score_targets, lengths in train_loader:
            batch_size = done_targets.shape[1]

            for key in features:
                features[key] = features[key].to(device)
            done_targets = done_targets.to(device)
            score_targets = score_targets.to(device)

            states = represent(features)  # (seq_len, batch, represent)

            packed_states = pack_padded_sequence(states, lengths,
                                                 enforce_sorted=False)
            h_0 = torch.zeros(1, batch_size, states.shape[-1], device=device)
            next_states, h_n = rnn(packed_states, h_0)  # (seq_len, batch, *)
            next_states, lengths = pad_packed_sequence(next_states)

            done_preds = is_done(next_states)  # (seq_len, batch, 1)

            score_preds = predict(h_n[-1])  # (batch, 2)

            masked_state_preds = apply_mask(next_states[:-1], lengths - 1,
                                            device)
            masked_state_targets = apply_mask(states[1:].detach(), lengths - 1,
                                              device)
            masked_done_preds = apply_mask(done_preds, lengths, device)
            masked_done_targets = apply_mask(done_targets, lengths, device)

            state_loss = cos_loss(masked_state_preds, masked_state_targets,
                                  torch.tensor(1, device=device))
            done_loss = bce_loss(masked_done_preds, masked_done_targets)
            score_loss = mse_loss(score_preds, score_targets)
            loss = state_loss * args.state_loss_coef + done_loss + score_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_state_loss += state_loss.item() * batch_size
            sum_done_loss += done_loss.item() * batch_size
            sum_score_loss += score_loss.item() * batch_size
            sum_loss += loss.item() * batch_size

        tb.add_scalar('train state loss',
                      sum_state_loss / len(train_loader.dataset), epoch)
        tb.add_scalar('train done loss',
                      sum_done_loss / len(train_loader.dataset), epoch)
        tb.add_scalar('train score loss',
                      sum_score_loss / len(train_loader.dataset), epoch)
        tb.add_scalar('train loss',
                      sum_loss / len(train_loader.dataset), epoch)

        represent.eval()
        is_done.eval()
        predict.eval()
        rnn.eval()
        sum_state_loss, sum_done_loss, sum_score_loss, sum_loss = 0, 0, 0, 0

        with torch.no_grad():
            for features, done_targets, score_targets, lengths in valid_loader:
                batch_size = done_targets.shape[1]

                for key in features:
                    features[key] = features[key].to(device)
                done_targets = done_targets.to(device)
                score_targets = score_targets.to(device)

                states = represent(features)  # (seq_len, batch, represent)

                packed_states = pack_padded_sequence(states, lengths,
                                                     enforce_sorted=False)
                h_0 = torch.zeros(1, batch_size, states.shape[-1],
                                  device=device)
                next_states, h_n = rnn(packed_states, h_0)  # (seq, batch, *)
                next_states, lengths = pad_packed_sequence(next_states)

                done_preds = is_done(next_states)  # (seq_len, batch, 1)

                score_preds = predict(h_n[-1])  # (batch, 2)

                masked_state_preds = apply_mask(next_states[:-1], lengths - 1,
                                                device)
                masked_state_targets = apply_mask(states[1:].detach(),
                                                  lengths - 1, device)
                masked_done_preds = apply_mask(done_preds, lengths, device)
                masked_done_targets = apply_mask(done_targets, lengths, device)

                state_loss = cos_loss(masked_state_preds, masked_state_targets,
                                      torch.tensor(1, device=device))
                done_loss = bce_loss(masked_done_preds, masked_done_targets)
                score_loss = mse_loss(score_preds, score_targets)
                loss = state_loss * args.state_loss_coef + done_loss + \
                    score_loss

                sum_state_loss += state_loss.item() * batch_size
                sum_done_loss += done_loss.item() * batch_size
                sum_score_loss += score_loss.item() * batch_size
                sum_loss += loss.item() * batch_size

            tb.add_scalar('valid state loss',
                          sum_state_loss / len(valid_loader.dataset), epoch)
            tb.add_scalar('valid done loss',
                          sum_done_loss / len(valid_loader.dataset), epoch)
            tb.add_scalar('valid score loss',
                          sum_score_loss / len(valid_loader.dataset), epoch)
            tb.add_scalar('valid loss',
                          sum_loss / len(valid_loader.dataset), epoch)

            for features, done_targets, score_targets, lengths in valid_loader:
                batch_size = done_targets.shape[1]

                first_features = {}
                for key in features:
                    first_features[key] = features[key][:1]
                first_done_targets = done_targets[:1]

                for key in first_features:
                    first_features[key] = first_features[key].to(device)
                first_done_targets = first_done_targets.to(device)
                score_targets = score_targets.to(device)

                state = represent(first_features)  # (1, batch, #features)
                states = [state]
                h_n = torch.zeros(1, batch_size, state.shape[-1],
                                  device=device)
                for _ in range(args.max_seq_len):
                    state, h_n = rnn(state, h_n)  # (seq_len, batch, *)
                    states.append(state)

                # Find first dones
                done_idx = [0] * batch_size
                for i, state in enumerate(states[1:], 1):
                    dones = is_done(state[0])
                    dones = dones.reshape(-1).tolist()
                    for j, done in enumerate(dones):
                        if done_idx[j] == 0 and done > 0.5:
                            done_idx[j] = i

                print(done_idx)

    tb.close()


if __name__ == '__main__':
    main()
