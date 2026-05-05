import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset

from Project3.src.utils.config import SEED
from Project3.src.utils.ELBO import ELBO

from Project3.src.GraphVAE import GraphVAE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

rng = torch.Generator().manual_seed(SEED)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
model = GraphVAE(in_channels=7, hidden_channels=32, latent_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():

    return

if __name__ == "__main__":
    train()
    