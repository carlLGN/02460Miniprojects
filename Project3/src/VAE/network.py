import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import random
import networkx as nx

from Project3.src.utils.config import SEED
from Project3.src.utils.ELBO import ELBO
from Project3.src.utils.message_parsing import AGGREGATE, UPDATE

class MPNNLayer(torch.nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__()

        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        message = AGGREGATE(x, edge_index)

        h = UPDATE(x, message)

        return self.linear(h)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = MPNNLayer(in_channels, hidden_channels)
        self.conv2 = MPNNLayer(hidden_channels, hidden_channels)


        self.mu_head = torch.nn.Linear(hidden_channels, latent_dim)
        self.logvar_head = torch.nn.Linear(hidden_channels, latent_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = self.conv1(x, edge_index).relu() #pass message to neighbour.
        h = self.conv2(h, edge_index).relu() #pass summaries to neighbors.
        
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()

        self.feature_reconstruction = torch.nn.Linear(latent_dim, out_channels)

    def forward(self, z, edge_index):
        
        edge_logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        x_hat_logits = self.feature_reconstruction(z)

        return edge_logits, x_hat_logits
    




class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(GraphVAE, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=in_channels)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)

        edge_logits, x_hat = self.decoder(z, data.edge_index)

        return edge_logits, x_hat, mu, logvar

