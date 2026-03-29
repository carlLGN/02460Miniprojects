import torch
import torch.nn as nn
import torch.optim as optim

from Project2.src.geodesics.curve_energy import curve_energy

def compute_geodesic_discrete(z_start, z_end, decoder, n_points=20, epochs=500, lr=0.01):
    """
    Optimizes a sequence of points in latent space to find the geodesic.
    
    Parameters:
    z_start: [M] tensor, starting point in latent space
    z_end: [M] tensor, ending point in latent space
    decoders: nn.ModuleList, list of VAE decoders
    n_points: int, number of discrete points along the curve
    epochs: int, number of optimization steps
    lr: float, learning rate
    """
    # straight line
    t = torch.linspace(0, 1, n_points, device=z_start.device).unsqueeze(1)
    initial_curve = z_start.unsqueeze(0) * (1 - t) + z_end.unsqueeze(0) * t
    interior_points = nn.Parameter(initial_curve[1:-1].detach().clone())
    
    optimizer = optim.Adam([interior_points], lr=lr)
    
    for epoch in range(epochs): #Energy minimizing learning
        optimizer.zero_grad()
        
        curve = torch.cat([z_start.unsqueeze(0), interior_points, z_end.unsqueeze(0)], dim=0)
        energy = curve_energy(curve, decoder)

        energy.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Energy: {energy.item():.4f}")
            
    final_curve = torch.cat([z_start.unsqueeze(0), interior_points, z_end.unsqueeze(0)], dim=0)
    return final_curve.detach()