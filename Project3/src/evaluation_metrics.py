import torch
import networkx as nx
import numpy as np
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# ---------------------------------------------------------
# 1. Load the Data & Baseline
# ---------------------------------------------------------
device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

try:
    baseline_matrices = torch.load('Project3/models/baseline_graphs.pt')
except FileNotFoundError:
    print("Error: 'Project3/models/baseline_graphs.pt' not found. Run your generation script first!")
    exit()

# ---------------------------------------------------------
# 2. Sanity Check Function (From Previous Step)
# ---------------------------------------------------------
def sanity_check_baseline(train_dataset, generated_matrices):
    """Performs basic structural sanity checks on the generated baseline graphs."""
    print("\n--- Baseline Sanity Checks ---")
    num_graphs = len(generated_matrices)
    print(f"1. Count Check: {num_graphs} graphs generated {'✅' if num_graphs == 1000 else '❌'}")
    
    is_symmetric, has_self_loops, is_binary = True, False, True
    total_generated_nodes, total_generated_edges = 0, 0

    for A_tensor in generated_matrices:
        A = A_tensor.numpy()
        if not np.allclose(A, A.T): is_symmetric = False
        if np.any(np.diag(A) != 0): has_self_loops = True
        if not np.array_equal(A, A.astype(bool)): is_binary = False
            
        total_generated_nodes += A.shape[0]
        total_generated_edges += np.sum(A) / 2

    print(f"2. Symmetry Check (Undirected): {'✅' if is_symmetric else '❌'}")
    print(f"3. Binary Adjacency Check: {'✅' if is_binary else '❌'}")
    print(f"4. Self-Loop Check (Should be False): {'❌' if has_self_loops else '✅'}")

# ---------------------------------------------------------
# 3. Part 2.4: Novelty and Uniqueness Evaluation
# ---------------------------------------------------------
def get_wl_hashes_from_dataset(pyg_dataset):
    """Hashes PyG dataset graphs using Weisfeiler-Lehman."""
    hashes = set()
    for data in pyg_dataset:
        G = to_networkx(data, to_undirected=True)
        hashes.add(nx.weisfeiler_lehman_graph_hash(G))
    return hashes

def get_wl_hashes_from_matrices(matrices):
    """Hashes a list of adjacency matrices using Weisfeiler-Lehman."""
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
    Computes metrics and prints the table required in Part 2.4.
    If generative_matrices is None, it only prints the baseline row.
    """
    print("\n--- Part 2.4: Evaluation Metrics Table ---")
    
    # 1. Get training hashes (ground truth)
    train_hashes = get_wl_hashes_from_dataset(train_dataset)
    
    # 2. Evaluate Baseline
    baseline_hashes = get_wl_hashes_from_matrices(baseline_matrices)
    b_novel, b_unique, b_novel_unique = calculate_metrics(baseline_hashes, train_hashes)
    
    # 3. Print the Table
    # Recreating the exact layout requested in the PDF
    print(f"{'':<25} | {'Novel':<10} | {'Unique':<10} | {'Novel+unique':<15}")
    print("-" * 65)
    print(f"{'Baseline':<25} | {b_novel:>5.0f}%     | {b_unique:>5.0f}%     | {b_novel_unique:>5.0f}%")
    
    # 4. Evaluate Deep Generative Model (If provided)
    if generative_matrices is not None:
        gen_hashes = get_wl_hashes_from_matrices(generative_matrices)
        g_novel, g_unique, g_novel_unique = calculate_metrics(gen_hashes, train_hashes)
        print(f"{'Deep generative model':<25} | {g_novel:>5.0f}%     | {g_unique:>5.0f}%     | {g_novel_unique:>5.0f}%")
    else:
        print(f"{'Deep generative model':<25} | {'N/A':>6}     | {'N/A':>6}     | {'N/A':>6}")

# =========================================================
# EXECUTION
# =========================================================
# Run Sanity Checks
#sanity_check_baseline(train_dataset, baseline_matrices)

# Run Part 2.4 Table Evaluation (Currently testing baseline only)
evaluate_and_print_table(train_dataset, baseline_matrices, generative_matrices=None)

# NOTE: When your group is ready, just load their saved matrices and do this:
# deep_model_matrices = torch.load('generative_graphs.pt')
# evaluate_and_print_table(train_dataset, baseline_matrices, generative_matrices=deep_model_matrices)