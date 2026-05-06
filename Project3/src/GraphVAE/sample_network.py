import torch
import random
import networkx as nx


def get_random_n(train_dataset):
    node_counts = [data.num_nodes for data in train_dataset]
    return random.choice(node_counts)

@torch.no_grad()
def sample_graphs(device, model, num_samples=1000, latent_dim=16, threshold=0.5):
    model.eval()
    generated_graphs = []

    for _ in range(num_samples):
        n = get_random_n()
        z = torch.randn(n, latent_dim).to(device)

        adj_logits = torch.matmul(z, z.t())
        adj_probs = torch.sigmoid(adj_logits)
        adj_matrix = (adj_probs > threshold).float()
        adj_matrix.fill_diagonal_(0)

        G = nx.from_numpy_array(adj_matrix.cpu().numpy())
        generated_graphs.append(G)

    return generated_graphs