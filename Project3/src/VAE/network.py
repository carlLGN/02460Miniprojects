import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import random
import networkx as nx

from Project3.src.utils.config import SEED
from Project3.src.utils.ELBO import ELBO
from Project3.src.utils.message_parsing import AGGREGATE, UPDATE

class MPNN(torch.nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__()

        self.linear = torch.nn.Linear(in_channels, out_channels)
        return

    def forward(self, x, edge_index):
        message = AGGREGATE(x, edge_index)

        h = UPDATE(x, message)

        return self.linear(h)

class encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(encoder, self).__init__()
        return

    def forward(self, data):
        return

class decoder(torch.nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(decoder, self).__init__()
        return

    def forward(self, z):
        return
    

