import argparse
import torch
import networkx as nx
import numpy as np
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

_parser = argparse.ArgumentParser()
_parser.add_argument('--model', choices=['node', 'graph'], default='graph',
                     help='Which deep model artifact to evaluate.')
_args = _parser.parse_args()
GEN_PATH = ('Project3/models/generative_graphs.pt' if _args.model == 'node'
            else 'Project3/models/generative_graphs_graph_level.pt')


device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

try:
    baseline_matrices = torch.load('Project3/models/baseline_graphs.pt', weights_only=False)
except FileNotFoundError:
    print("Error: 'Project3/models/baseline_graphs.pt' not found. Run your generation script first!")
    exit()

try:
    generative_matrices = torch.load(GEN_PATH, weights_only=False)
    print(f"Loaded deep generative graphs from {GEN_PATH}")
except FileNotFoundError:
    print(f"Error: '{GEN_PATH}' not found. Run train_network.py with the matching --model first!")
    exit()


def sanity_check_baseline(train_dataset, generated_matrices):
    print("\n--- Baseline Sanity Checks ---")
    num_graphs = len(generated_matrices)
    print(f"1. Count Check: {num_graphs} graphs generated {'True' if num_graphs == 1000 else 'False'}")
    
    is_symmetric, has_self_loops, is_binary = True, False, True
    total_generated_nodes, total_generated_edges = 0, 0

    for A_tensor in generated_matrices:
        A = A_tensor.numpy()
        if not np.allclose(A, A.T): is_symmetric = False
        if np.any(np.diag(A) != 0): has_self_loops = True
        if not np.array_equal(A, A.astype(bool)): is_binary = False
            
        total_generated_nodes += A.shape[0]
        total_generated_edges += np.sum(A) / 2

    print(f"2. Symmetry Check (Undirected): {'True' if is_symmetric else 'False'}")
    print(f"3. Binary Adjacency Check: {'True' if is_binary else 'False'}")
    print(f"4. Self-Loop Check (Should be False): {'False' if has_self_loops else 'True'}")


def get_wl_hashes_from_dataset(pyg_dataset):
    hashes = set()
    for data in pyg_dataset:
        G = to_networkx(data, to_undirected=True)
        hashes.add(nx.weisfeiler_lehman_graph_hash(G))
    return hashes

def get_wl_hashes_from_matrices(matrices):
    hashes = []
    for A_tensor in matrices:
        G = nx.from_numpy_array(A_tensor.numpy())
        hashes.append(nx.weisfeiler_lehman_graph_hash(G))
    return hashes

def calculate_metrics(generated_hashes, train_hashes):
    """Calculates Novelty, Uniqueness, and Novel+Unique percentages."""
    total = len(generated_hashes)
    if total == 0:
        return 0.0, 0.0, 0.0
        
    unique_hashes = set(generated_hashes)
    unique_pct = (len(unique_hashes) / total) * 100
    
    novel_count = sum(1 for h in generated_hashes if h not in train_hashes)
    novel_pct = (novel_count / total) * 100
    
    novel_and_unique_hashes = unique_hashes - train_hashes
    novel_unique_pct = (len(novel_and_unique_hashes) / total) * 100
    
    return novel_pct, unique_pct, novel_unique_pct

def evaluate_and_print_table(train_dataset, baseline_matrices, generative_matrices=None):
    """
    Computes metrics and prints table.
    If generative_matrices is None, it only prints the baseline row.
    """
    print("\n--- Evaluation Metrics Table ---")
    
    train_hashes = get_wl_hashes_from_dataset(train_dataset)
    
    baseline_hashes = get_wl_hashes_from_matrices(baseline_matrices)
    b_novel, b_unique, b_novel_unique = calculate_metrics(baseline_hashes, train_hashes)
    
    print(f"{'':<25} | {'Novel':<10} | {'Unique':<10} | {'Novel+unique':<15}")
    print("-" * 65)
    print(f"{'Baseline':<25} | {b_novel:>5.0f}%     | {b_unique:>5.0f}%     | {b_novel_unique:>5.0f}%")
    
    if generative_matrices is not None:
        gen_hashes = get_wl_hashes_from_matrices(generative_matrices)
        g_novel, g_unique, g_novel_unique = calculate_metrics(gen_hashes, train_hashes)
        print(f"{'Deep generative model':<25} | {g_novel:>5.0f}%     | {g_unique:>5.0f}%     | {g_novel_unique:>5.0f}%")
    else:
        print(f"{'Deep generative model':<25} | {'N/A':>6}     | {'N/A':>6}     | {'N/A':>6}")


# run sanity checks if required
#sanity_check_baseline(train_dataset, baseline_matrices)

evaluate_and_print_table(train_dataset, baseline_matrices, generative_matrices=generative_matrices)