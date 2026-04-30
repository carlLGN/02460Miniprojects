import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# ---------------------------------------------------------
# 1. Load the Data
# ---------------------------------------------------------
device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Load the generated baseline graphs
try:
    baseline_matrices = torch.load('Project3/models/baseline_graphs.pt')
    print("Loaded baseline graphs.")
except FileNotFoundError:
    print("Error: 'Project3/models/baseline_graphs.pt' not found. Please run the generation script first!")
    exit()

# ---------------------------------------------------------
# 2. Extract Graph Statistics
# ---------------------------------------------------------
def compute_graph_statistics(graphs, is_pyg_dataset=False):
    """
    Computes Node Degree, Clustering Coefficient, and Eigenvector Centrality 
    for all nodes across a list of graphs.
    """
    degrees = []
    clustering_coeffs = []
    eigenvector_centralities = []
    
    print(f"Computing statistics for {len(graphs)} graphs. This might take a few seconds...")
    
    for item in graphs:
        # Convert to NetworkX format
        if is_pyg_dataset:
            G = to_networkx(item, to_undirected=True)
        else:
            G = nx.from_numpy_array(item.numpy())
        
        # 1. Node Degrees
        deg = [d for n, d in G.degree()]
        degrees.extend(deg)
        
        # 2. Clustering Coefficients
        cc = list(nx.clustering(G).values())
        clustering_coeffs.extend(cc)
        
        # 3. Eigenvector Centrality
        # Note: Erdös-Rényi baseline often generates disconnected graphs. 
        # The standard power iteration method in NetworkX can fail to converge on these.
        # We use a try-except block to fall back to a more robust NumPy solver if it fails.
        try:
            ec = list(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-3).values())
        except nx.PowerIterationFailedConvergence:
            try:
                ec = list(nx.eigenvector_centrality_numpy(G).values())
            except Exception:
                # Extreme fallback for completely degenerate/empty graphs
                ec = [0.0] * len(G.nodes)
                
        eigenvector_centralities.extend(ec)
        
    return {
        'degree': degrees,
        'clustering': clustering_coeffs,
        'eigenvector': eigenvector_centralities
    }

# ---------------------------------------------------------
# 3. Plot the 3x3 Histograms
# ---------------------------------------------------------
def plot_3x3_histograms(train_stats, baseline_stats, gen_stats=None, save_path='histograms.png'):
    """
    Plots the 3x3 grid ensuring exactly identical bins across each row as required.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    metrics = ['degree', 'clustering', 'eigenvector']
    metric_names = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']
    column_titles = ['Training Data (Empirical)', 'Baseline (Erdös-Rényi)', 'Deep Generative Model']
    
    for i, metric in enumerate(metrics):
        # Pool all available data together to find the global min and max
        all_data = train_stats[metric] + baseline_stats[metric]
        if gen_stats is not None:
            all_data += gen_stats[metric]
            
        # Define universal bins for this specific metric
        if metric == 'degree':
            # Node degree is discrete, so we use integer binning centered on whole numbers
            min_val, max_val = int(min(all_data)), int(max(all_data))
            bins = np.arange(min_val, max_val + 2) - 0.5 
        else:
            # Clustering and Centrality are continuous (0.0 to 1.0)
            bins = np.linspace(min(all_data), max(all_data), 20)
            
        # Plot 1: Training Data
        axes[i, 0].hist(train_stats[metric], bins=bins, color='skyblue', edgecolor='black')
        axes[i, 0].set_ylabel(metric_names[i], fontsize=14, fontweight='bold')
        
        # Plot 2: Baseline
        axes[i, 1].hist(baseline_stats[metric], bins=bins, color='salmon', edgecolor='black')
        
        # Plot 3: Generative Model
        if gen_stats is not None:
            axes[i, 2].hist(gen_stats[metric], bins=bins, color='lightgreen', edgecolor='black')
        else:
            # Draw placeholder text if generative data isn't ready
            axes[i, 2].text(0.5, 0.5, 'Generative Model Data\nNot Provided Yet', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[i, 2].transAxes, fontsize=12, color='gray')
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            
        # Set column titles only on the top row
        if i == 0:
            axes[i, 0].set_title(column_titles[0], fontsize=15)
            axes[i, 1].set_title(column_titles[1], fontsize=15)
            axes[i, 2].set_title(column_titles[2], fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot successfully generated and saved to: {save_path}")

# =========================================================
# EXECUTION
# =========================================================
# 1. Compute stats for Training set
print("Processing Training Data...")
train_stats = compute_graph_statistics(train_dataset, is_pyg_dataset=True)

# 2. Compute stats for Baseline
print("Processing Baseline Data...")
baseline_stats = compute_graph_statistics(baseline_matrices, is_pyg_dataset=False)

# 3. Plot and save
plot_3x3_histograms(train_stats, baseline_stats, gen_stats=None, save_path='project_histograms.png')


# NOTE FOR LATER: When your group finishes the generative model:
# 1. Load their matrices: 
#    generative_matrices = torch.load('generative_graphs.pt')
# 2. Compute stats: 
#    gen_stats = compute_graph_statistics(generative_matrices, is_pyg_dataset=False)
# 3. Pass to plotting function:
#    plot_3x3_histograms(train_stats, baseline_stats, gen_stats=gen_stats)