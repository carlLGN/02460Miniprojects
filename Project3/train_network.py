import torch
import argparse
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from Project3.src.utils.config import SEED
from Project3.src.utils.ELBO import ELBO

from Project3.src.GraphVAE.network import GraphVAE

def main():
    parser = argparse.ArgumentParser(description='GraphVAE Training on MUTAG')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden channels in MPNN')
    parser.add_argument('--latent', type=int, default=16, help='Dimension of node latents')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)

    rng = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, (100, 44, 44), generator=rng)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = GraphVAE(
        in_channels=dataset.num_node_features, 
        hidden_channels=args.hidden, 
        latent_dim=args.latent
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Loss: {train_loss:.4f}')

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to_device()
        optimizer.zero_grad()

        edge_logits, x_hat, mu, logvar = model(data)

        neg_edge_index = negative_sampling( 
            edge_index=data.edge_index, 
            num_nodes=data.num_nodes
        )
        neg_edge_logits, _ = model.decoder(model.reparameterize(mu, logvar), neg_edge_index)

        loss = ELBO(edge_logits,neg_edge_logits, x_hat, data.x, mu, logvar )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        return total_loss / len(loader)


if __name__ == "__main__":
    main()
    
#TODO: val og test