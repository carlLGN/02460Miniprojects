import torch
import torch.nn as nn
import torch.distributions as td

def curve_energy(curve, decoders):
    """
    Model average curve energy. Functions as normal curve energy for 1 decoder.

    curve: [N, M] tensor of points in latent space
    decoders: nn.ModuleList of decoder models
    """
    N = curve.shape[0]
    n_decoders = len(decoders)
    energy = 0
    
    for i in range(N - 1):
        # sample two random decoders l and k
        l = torch.randint(0, n_decoders, (1,)).item()
        k = torch.randint(0, n_decoders, (1,)).item()
        
        z_i = curve[i]
        z_next = curve[i+1]
        
        f_l_zi = decoders[l](z_i.unsqueeze(0)).mean
        f_k_znext = decoders[k](z_next.unsqueeze(0)).mean
        
        energy += torch.sum((f_l_zi - f_k_znext) ** 2)
        
    return energy

def standard_curve_energy(curve, decoder):
    """
    Standard curve energy
    
    Parameters:
    curve: [N, M] tensor of points in latent space
    decoder: A single PyTorch nn.Module
    """

    decoder = decoder[0]
    mu = decoder(curve).mean 
    energy = torch.sum((mu[:-1] - mu[1:]) ** 2)
    
    return energy