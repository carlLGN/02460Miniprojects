import matplotlib.pyplot as plt
import torch

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from Project2.src.geodesics.compute_geodesics import compute_geodesic_discrete

def plot_latent_geodesics(z_points, labels, decoders, n_pairs=25,fixed_pairs=None):

    z_vis = z_points.detach().cpu().numpy()


    #Plotting code: made by Gemini
    plt.figure(figsize=(12, 10))
    
    # We use 'tab10' colormap, which is great for distinct categorical classes (like MNIST digits)
    scatter = plt.scatter(z_vis[:, 0], z_vis[:, 1], c=labels, cmap='tab10', alpha=0.3, s=15, zorder=1)

    handles, class_labels = scatter.legend_elements()
    plt.legend(handles, class_labels, title="Data Classes", loc="best")
    # Select random pairs 
    num_points = z_points.shape[0]

    if fixed_pairs is not None:
        start_indices = [pair[0] for pair in fixed_pairs]
        end_indices = [pair[1] for pair in fixed_pairs]
        n_pairs_to_plot = len(fixed_pairs)
    else:
        np.random.seed(42)
        start_indices = np.random.choice(num_points, n_pairs, replace=False)
        end_indices = np.random.choice(num_points, n_pairs, replace=False)
        n_pairs_to_plot = n_pairs
    
    # Compute and plot paths 
    for i in range(n_pairs_to_plot):

        z_start = z_points[start_indices[i]]
        z_end = z_points[end_indices[i]]
        
        geo_path = compute_geodesic_discrete(z_start, z_end, decoders) 
        
        # Convert path to numpy
        geo_np = geo_path.cpu().numpy()
        z_s_np = z_start.unsqueeze(0).cpu().numpy()
        z_e_np = z_end.unsqueeze(0).cpu().numpy()
        
        geo_vis = geo_np
        z_s_vis = z_s_np[0]
        z_e_vis = z_e_np[0]
            
        # euclidean straight line for comparison.
        # plt.plot([z_s_vis[0], z_e_vis[0]], [z_s_vis[1], z_e_vis[1]], 
        #          color='gray', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
        
        # Plot Riemannian Geodesic curve (black, solid)
        plt.plot(geo_vis[:, 0], geo_vis[:, 1], color='black', linewidth=2.5, zorder=3)
        
        # Plot start and end markers (red dots)
        plt.scatter([z_s_vis[0], z_e_vis[0]], [z_s_vis[1], z_e_vis[1]], color='red', s=30, zorder=4)

    # Formatting
    plt.title("Latent Space Geometry: Euclidean (Dashed) vs. Geodesic (Solid)", fontsize=16)
    plt.xlabel("Latent Dim 1", fontsize=12)
    plt.ylabel("Latent Dim 2", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("./Project2/outputs/geodesics.png", dpi=300, bbox_inches='tight')
    plt.show()

