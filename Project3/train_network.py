import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset

from Project3.src.utils.config import SEED
from Project3.src.utils.ELBO import ELBO

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

rng = torch.Generator().manual_seed(SEED)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


def train():

    return

if __name__ == "__main__":
    train()
    