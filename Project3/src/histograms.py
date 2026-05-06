import argparse
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

_parser = argparse.ArgumentParser()
_parser.add_argument('--model', choices=['node', 'graph'], default='graph',
                     help='Which deep model artifact to plot.')
_args = _parser.parse_args()
GEN_PATH = ('Project3/models/generative_graphs.pt' if _args.model == 'node'
            else 'Project3/models/generative_graphs_graph_level.pt')


device = 'cpu'
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


# loads graphs if baseline and generative have been run
try:
    baseline_matrices = torch.load('Project3/models/baseline_graphs.pt', weights_only=False)
    print("Loaded baseline graphs.")
except FileNotFoundError:
    print("Error: 'Project3/models/baseline_graphs.pt' not found. Please run the generation script first!")
    exit()

try:
    generative_matrices = torch.load(GEN_PATH, weights_only=False)
    print(f"Loaded deep generative graphs from {GEN_PATH}")
except FileNotFoundError:
    print(f"Error: '{GEN_PATH}' not found. Please run train_network.py with the matching --model first!")
    exit()


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
        if is_pyg_dataset:
            G = to_networkx(item, to_undirected=True)
        else:
            G = nx.from_numpy_array(item.numpy())
        
        deg = [d for n, d in G.degree()]
        degrees.extend(deg)
        
        cc = list(nx.clustering(G).values())
        clustering_coeffs.extend(cc)
        
        # use try-expect block as Erdös-Rényi baseline can generate disconnected graphs which networkX can fail to converge on
        # fall back to numpy solver in case of fail
        try:
            ec = list(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-3).values())
        except nx.PowerIterationFailedConvergence:
            try:
                ec = list(nx.eigenvector_centrality_numpy(G).values())
            except Exception:
                ec = [0.0] * len(G.nodes)
                
        eigenvector_centralities.extend(ec)
        
    return {
        'degree': degrees,
        'clustering': clustering_coeffs,
        'eigenvector': eigenvector_centralities
    }


def plot_3x3_histograms(train_stats, baseline_stats, gen_stats=None, save_path='histograms.png'):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    metrics = ['degree', 'clustering', 'eigenvector']
    metric_names = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']
    column_titles = ['Training Data (Empirical)', 'Baseline (Erdös-Rényi)', 'Deep Generative Model']
    
    for i, metric in enumerate(metrics):
        all_data = train_stats[metric] + baseline_stats[metric]
        if gen_stats is not None:
            all_data += gen_stats[metric]
            
        if metric == 'degree':
            min_val, max_val = int(min(all_data)), int(max(all_data))
            bins = np.arange(min_val, max_val + 2) - 0.5 
        else:
            bins = np.linspace(min(all_data), max(all_data), 20)
            
        axes[i, 0].hist(train_stats[metric], bins=bins, color='skyblue', edgecolor='black')
        axes[i, 0].set_ylabel(metric_names[i], fontsize=14, fontweight='bold')
        
        axes[i, 1].hist(baseline_stats[metric], bins=bins, color='salmon', edgecolor='black')
        
        if gen_stats is not None:
            axes[i, 2].hist(gen_stats[metric], bins=bins, color='lightgreen', edgecolor='black')
        else:
            axes[i, 2].text(0.5, 0.5, 'Generative Model Data\nNot Provided Yet', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[i, 2].transAxes, fontsize=12, color='gray')
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            
        if i == 0:
            axes[i, 0].set_title(column_titles[0], fontsize=15)
            axes[i, 1].set_title(column_titles[1], fontsize=15)
            axes[i, 2].set_title(column_titles[2], fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot successfully generated and saved to: {save_path}")


print("Processing Training Data...")
train_stats = compute_graph_statistics(train_dataset, is_pyg_dataset=True)

print("Processing Baseline Data...")
baseline_stats = compute_graph_statistics(baseline_matrices, is_pyg_dataset=False)

print("Processing Deep Generative Model Data...")
gen_stats = compute_graph_statistics(generative_matrices, is_pyg_dataset=False)

plot_3x3_histograms(train_stats, baseline_stats, gen_stats=gen_stats, save_path='project_histograms.png')