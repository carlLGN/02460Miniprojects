"""Graph-level latent VAE for MUTAG.

One latent z_G per graph (vs one per node in the original GraphVAE). The decoder
generates all edges of an n-node graph jointly, conditioned on z_G plus a
learned positional embedding for each node, so global structure (connectivity,
clustering) can be captured in z_G.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from Project3.src.GraphVAE.network import MPNNLayer


MAX_NODES = 28  # MUTAG max graph size; learned positional embedding capacity


class GraphLevelEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = MPNNLayer(in_channels, hidden_channels)
        self.conv2 = MPNNLayer(hidden_channels, hidden_channels)
        self.mu_head = nn.Linear(hidden_channels, latent_dim)
        self.logvar_head = nn.Linear(hidden_channels, latent_dim)

    def forward(self, data):
        h = self.conv1(data.x, data.edge_index).relu()
        h = self.conv2(h, data.edge_index).relu()
        graph_h = global_mean_pool(h, data.batch)  # (B, hidden)
        return self.mu_head(graph_h), self.logvar_head(graph_h)


class GraphLevelDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, max_nodes=MAX_NODES, pos_dim=16):
        super().__init__()
        self.max_nodes = max_nodes
        self.pos_embedding = nn.Embedding(max_nodes, pos_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim + pos_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.edge_bias = nn.Parameter(torch.zeros(1))

    def forward(self, z, n_per_graph):
        """Decode batch of graphs.

        Args:
            z: (B, latent_dim)
            n_per_graph: (B,) long tensor of node counts

        Returns:
            edge_logits: (B, max_n, max_n)  symmetric pairwise logits
            mask:        (B, max_n, max_n)  bool, True only for upper-tri pairs
                         within each graph's n (so loss is over valid edges)
        """
        device = z.device
        B = z.size(0)
        max_n = int(n_per_graph.max().item())

        idx = torch.arange(max_n, device=device)
        pos = self.pos_embedding(idx)  # (max_n, pos_dim)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, max_n, pos_dim)
        z_exp = z.unsqueeze(1).expand(-1, max_n, -1)  # (B, max_n, latent)
        h = self.node_mlp(torch.cat([z_exp, pos], dim=-1))  # (B, max_n, hidden)

        # Symmetric pairwise: h_i + h_j is symmetric in (i, j).
        h_i = h.unsqueeze(2)  # (B, max_n, 1, hidden)
        h_j = h.unsqueeze(1)  # (B, 1, max_n, hidden)
        edge_logits = self.edge_mlp(h_i + h_j).squeeze(-1) + self.edge_bias

        # Build per-graph upper-triangular mask
        mask = torch.zeros(B, max_n, max_n, dtype=torch.bool, device=device)
        upper_max = torch.triu(torch.ones(max_n, max_n, dtype=torch.bool, device=device), diagonal=1)
        for i, n in enumerate(n_per_graph.tolist()):
            mask[i, :n, :n] = upper_max[:n, :n]

        return edge_logits, mask


class GraphLevelVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, max_nodes=MAX_NODES):
        super().__init__()
        self.encoder = GraphLevelEncoder(in_channels, hidden_channels, latent_dim)
        self.decoder = GraphLevelDecoder(latent_dim, hidden_channels, max_nodes=max_nodes)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, data, n_per_graph):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        edge_logits, mask = self.decoder(z, n_per_graph)
        return edge_logits, mask, mu, logvar
