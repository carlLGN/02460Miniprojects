import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import random
import networkx as nx

from Project3.src.utils.config import SEED


device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

rng = torch.Generator().manual_seed(SEED)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


def generate_erdos_renyi_baseline(train_dataset, num_samples=1000):

    num_nodes_list = [data.num_nodes for data in train_dataset]
    
    density_per_N = {}
    unique_Ns = set(num_nodes_list)
    
    for N in unique_Ns:
        graphs_with_N_nodes = [data for data in train_dataset if data.num_nodes == N]
        
        total_undirected_edges = 0
        for data in graphs_with_N_nodes:
            total_undirected_edges += data.edge_index.shape[1] // 2
            
        if N > 1:
            total_possible_edges = len(graphs_with_N_nodes) * (N * (N - 1) / 2)
            r = total_undirected_edges / total_possible_edges
        else:
            r = 0.0
            
        density_per_N[N] = r

    generated_adjacency_matrices = []
    
    for _ in range(num_samples):
        N = random.choice(num_nodes_list)
        
        r = density_per_N[N]
        
        G = nx.erdos_renyi_graph(n=N, p=r)
        
        A = nx.to_numpy_array(G)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        
        generated_adjacency_matrices.append(A_tensor)
        
    return generated_adjacency_matrices


sampled_graphs = generate_erdos_renyi_baseline(train_dataset, num_samples=1000)

print(f"Successfully generated {len(sampled_graphs)} baseline graphs.")
print(f"Example shape of the first generated adjacency matrix: {sampled_graphs[0].shape}")

torch.save(sampled_graphs, 'Project3/models/baseline_graphs.pt')
print("Saved baseline graphs to 'Project3/models/baseline_graphs.pt'")