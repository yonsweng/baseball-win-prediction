import torch


class MCTS:
    def __init__(self, rank, args):
        self.device = torch.device(f'cuda:{rank % args.n_gpus}')

    def get_policy(self):
        pass

    def search(self):
        pass
