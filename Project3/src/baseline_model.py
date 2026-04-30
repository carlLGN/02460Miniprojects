import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import random
import networkx as nx

# ---------------------------------------------------------
# 1. Load Data (From your gnn_graph_classification.py)
# ---------------------------------------------------------
device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# ---------------------------------------------------------
# 2. Implement the Erdös-Rényi Baseline (Part 2.2)
# ---------------------------------------------------------
def generate_erdos_renyi_baseline(train_dataset, num_samples=1000):
    """
    Generates baseline graphs using the Erdös-Rényi model based on empirical data.
    """
    # Step 1: Extract the empirical distribution of N (number of nodes)
    # By keeping this as a list with repetitions, random.choice() will 
    # automatically sample from the empirical distribution.
    num_nodes_list = [data.num_nodes for data in train_dataset]
    
    # Step 2: Pre-calculate the link probability (density) 'r' for each unique N
    # This avoids recalculating the density every time we sample a graph.
    density_per_N = {}
    unique_Ns = set(num_nodes_list)
    
    for N in unique_Ns:
        # Find all training graphs that have exactly N nodes
        graphs_with_N_nodes = [data for data in train_dataset if data.num_nodes == N]
        
        total_undirected_edges = 0
        for data in graphs_with_N_nodes:
            # PyTorch Geometric stores undirected edges as two directed edges.
            # We divide by 2 to get the actual number of undirected edges.
            total_undirected_edges += data.edge_index.shape[1] // 2
            
        # Maximum possible undirected edges for all graphs with N nodes
        # Formula for one graph: N * (N - 1) / 2
        if N > 1:
            total_possible_edges = len(graphs_with_N_nodes) * (N * (N - 1) / 2)
            r = total_undirected_edges / total_possible_edges
        else:
            r = 0.0 # A graph with 1 node cannot have edges (assuming no self-loops)
            
        density_per_N[N] = r

    # Step 3: Sample the graphs
    generated_adjacency_matrices = []
    
    for _ in range(num_samples):
        # Sample N from the empirical distribution
        N = random.choice(num_nodes_list)
        
        # Get the corresponding edge probability r
        r = density_per_N[N]
        
        # Generate the Erdös-Rényi graph
        G = nx.erdos_renyi_graph(n=N, p=r)
        
        # The project requires generating graph adjacency matrices 
        A = nx.to_numpy_array(G)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        
        generated_adjacency_matrices.append(A_tensor)
        
    return generated_adjacency_matrices

# ---------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------
# Part 2.4 requires you to sample 1000 graphs from the baseline[cite: 41].
sampled_graphs = generate_erdos_renyi_baseline(train_dataset, num_samples=1000)

print(f"Successfully generated {len(sampled_graphs)} baseline graphs.")
print(f"Example shape of the first generated adjacency matrix: {sampled_graphs[0].shape}")