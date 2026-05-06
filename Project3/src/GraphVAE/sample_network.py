import torch
import random


def get_random_n(train_dataset):
    node_counts = [data.num_nodes for data in train_dataset]
    return random.choice(node_counts)


@torch.no_grad()
def sample_graphs(device, model, train_dataset, num_samples=1000, latent_dim=16, threshold=0.5):
    """Sample graphs from the prior p(z) = N(0, I) and return adjacency tensors.

    Returns a list of torch.float32 adjacency matrices (same format as baseline_graphs.pt).
    """
    model.eval()
    adjacency_matrices = []

    edge_bias = getattr(model.decoder, 'edge_bias', torch.zeros(1, device=device))

    for _ in range(num_samples):
        n = get_random_n(train_dataset)
        z = torch.randn(n, latent_dim, device=device)

        adj_logits = torch.matmul(z, z.t()) + edge_bias
        adj_probs = torch.sigmoid(adj_logits)
        adj_matrix = (adj_probs > threshold).float()

        # enforce symmetry and no self-loops via upper triangle
        upper = torch.triu(adj_matrix, diagonal=1)
        adj_matrix = upper + upper.t()

        adjacency_matrices.append(adj_matrix.cpu())

    return adjacency_matrices


@torch.no_grad()
def sample_graphs_graph_level(device, model, train_dataset, num_samples=1000,
                              latent_dim=32, threshold=0.5, bernoulli=False):
    """Sample graphs from the graph-level VAE prior p(z_G) = N(0, I).

    For each sample: draw n from the empirical node-count distribution, draw
    z_G ~ N(0, I), decode to an n x n upper-triangular logit matrix, and either
    threshold or Bernoulli-sample to get edges.
    """
    model.eval()
    adjacency_matrices = []

    for _ in range(num_samples):
        n = get_random_n(train_dataset)
        z = torch.randn(1, latent_dim, device=device)
        n_per_graph = torch.tensor([n], device=device, dtype=torch.long)

        edge_logits, mask = model.decoder(z, n_per_graph)
        # edge_logits: (1, n, n); mask is upper-tri within first n rows/cols
        probs = torch.sigmoid(edge_logits[0])  # (n, n)
        if bernoulli:
            sampled = torch.bernoulli(probs)
        else:
            sampled = (probs > threshold).float()

        upper = torch.triu(sampled, diagonal=1)
        adj_matrix = upper + upper.t()
        adjacency_matrices.append(adj_matrix.cpu())

    return adjacency_matrices
