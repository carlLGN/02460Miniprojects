import os
import torch
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, to_dense_adj

from Project3.src.utils.ELBO import ELBO, graph_level_ELBO

from Project3.src.GraphVAE.network import GraphVAE
from Project3.src.GraphVAE.graph_level_network import GraphLevelVAE, MAX_NODES
from Project3.src.GraphVAE.sample_network import sample_graphs, sample_graphs_graph_level


MODELS_DIR = 'Project3/models'


def main():
    parser = argparse.ArgumentParser(description='GraphVAE Training on MUTAG')
    parser.add_argument('--model', choices=['node', 'graph'], default='node',
                        help='node = per-node latent VAE; graph = graph-level latent VAE')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--latent', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Edge prob threshold. Defaults: node=0.9, graph=0.7.')
    parser.add_argument('--neg_ratio', type=int, default=5,
                        help='Negatives per positive edge (node-level model only)')
    parser.add_argument('--beta', type=float, default=None,
                        help='KL weight; default 0.1 for node, 1.0 for graph')
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight in BCE (graph-level model only). Defaults to (1-d)/d using training density.')
    parser.add_argument('--bernoulli', action='store_true',
                        help='Sample edges by Bernoulli instead of thresholding (graph-level only)')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)

    train_set = dataset

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    if args.model == 'node':
        model_path = os.path.join(MODELS_DIR, 'graph_vae.pt')
        graphs_path = os.path.join(MODELS_DIR, 'generative_graphs.pt')
        model = GraphVAE(dataset.num_node_features, args.hidden, args.latent).to(device)
        beta = 0.1 if args.beta is None else args.beta
        if args.threshold is None:
            args.threshold = 0.9
    else:
        model_path = os.path.join(MODELS_DIR, 'graph_level_vae.pt')
        graphs_path = os.path.join(MODELS_DIR, 'generative_graphs_graph_level.pt')
        model = GraphLevelVAE(dataset.num_node_features, args.hidden, args.latent,
                              max_nodes=MAX_NODES).to(device)
        beta = 1.0 if args.beta is None else args.beta
        if args.threshold is None:
            args.threshold = 0.7
        if args.pos_weight is None:
            density = sum(d.edge_index.shape[1] / max(1, d.num_nodes * (d.num_nodes - 1))
                          for d in train_set) / len(train_set)
            args.pos_weight = float((1 - density) / max(density, 1e-6))
            print(f'Auto pos_weight = {args.pos_weight:.2f}  (training density = {density:.3f})')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        if args.model == 'node':
            train_loss = train_node(model, train_loader, optimizer, device, args.neg_ratio, beta)
        else:
            train_loss = train_graph(model, train_loader, optimizer, device, beta, args.pos_weight)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Loss: {train_loss:.4f}')

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path}')

    if args.model == 'node':
        graphs = sample_graphs(device, model, train_set,
                               num_samples=args.num_samples,
                               latent_dim=args.latent,
                               threshold=args.threshold)
    else:
        graphs = sample_graphs_graph_level(device, model, train_set,
                                           num_samples=args.num_samples,
                                           latent_dim=args.latent,
                                           threshold=args.threshold,
                                           bernoulli=args.bernoulli)
    torch.save(graphs, graphs_path)
    print(f'Saved {len(graphs)} sampled graphs to {graphs_path}')


def train_node(model, loader, optimizer, device, neg_ratio, beta):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge_logits, x_hat, mu, logvar = model(data)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.shape[1] * neg_ratio,
        )
        neg_edge_logits, _ = model.decoder(model.reparameterize(mu, logvar), neg_edge_index)
        loss = ELBO(edge_logits, neg_edge_logits, x_hat, data.x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_graph(model, loader, optimizer, device, beta, pos_weight):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Recover per-graph node counts from the batch index
        n_per_graph = torch.bincount(data.batch)
        target_adj = to_dense_adj(data.edge_index, batch=data.batch)  # (B, max_n, max_n)

        edge_logits, mask, mu, logvar = model(data, n_per_graph)
        loss, _, _ = graph_level_ELBO(edge_logits, mask, target_adj, mu, logvar,
                                       beta=beta, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    main()
