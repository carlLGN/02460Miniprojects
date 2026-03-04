"""
Plot the prior p(z) and aggregate posterior q(z) = 1/N sum_i q(z|x_i)
for a trained VAE with a 2D latent space.

Supports Gaussian, Mixture of Gaussians (MoG), and Flow priors.

Usage:
    python plot_prior_posterior.py --model model.pt --prior gaussian
    python plot_prior_posterior.py --model model.pt --prior mix
    python plot_prior_posterior.py --model model.pt --prior flow

Created by Claude
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from Project1.src.vae.vae import build_model

import argparse


def compute_aggregate_posterior_density(model, data_loader, device, grid_z1, grid_z2):
    """
    Compute q(z) = 1/N sum_i q(z|x_i) on a 2D grid.
    Works identically for any prior since it only touches the encoder.
    """
    model.eval()
    zz1, zz2 = np.meshgrid(grid_z1, grid_z2)
    grid_points = torch.tensor(
        np.stack([zz1.ravel(), zz2.ravel()], axis=-1), dtype=torch.float32
    ).to(device)  # (G, 2)

    all_log_probs = []

    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            q = model.encoder(x_batch)

            loc = q.base_dist.loc      # (B, 2)
            scale = q.base_dist.scale   # (B, 2)

            diff = grid_points.unsqueeze(1) - loc.unsqueeze(0)       # (G, B, 2)
            log_p = -0.5 * ((diff / scale.unsqueeze(0)) ** 2) \
                    - scale.unsqueeze(0).log() \
                    - 0.5 * np.log(2 * np.pi)                        # (G, B, 2)
            log_p = log_p.sum(dim=-1)  # (G, B)

            all_log_probs.append(log_p.cpu())

    all_log_probs = torch.cat(all_log_probs, dim=1)  # (G, N)
    N = all_log_probs.shape[1]

    log_aggregate = torch.logsumexp(all_log_probs, dim=1) - np.log(N)
    density = log_aggregate.exp().numpy().reshape(zz1.shape)
    return zz1, zz2, density


def compute_prior_density(model, device, grid_z1, grid_z2):
    """
    Evaluate the prior p(z) on the grid.
    Uses model.prior().log_prob(z) which works uniformly for all three
    prior types (Gaussian, MoG, Flow) thanks to their shared interface.
    """
    model.eval()
    zz1, zz2 = np.meshgrid(grid_z1, grid_z2)
    grid_points = torch.tensor(
        np.stack([zz1.ravel(), zz2.ravel()], axis=-1), dtype=torch.float32
    ).to(device)  # (G, 2)

    with torch.no_grad():
        prior_dist = model.prior()
        log_p = prior_dist.log_prob(grid_points)  # (G,)

    density = log_p.cpu().exp().numpy().reshape(zz1.shape)
    return zz1, zz2, density


def infer_params_from_state_dict(state_dict, prior_type):
    """Recover hyperparameters from a saved state dict (mirrors vae.py logic)."""
    if prior_type == "gaussian":
        return {"latent_dim": state_dict["prior.mean"].shape[-1]}
    elif prior_type == "mix":
        return {
            "latent_dim": state_dict["prior.mean"].shape[-1],
            "K": state_dict["prior.mean"].shape[0],
        }
    elif prior_type == "flow":
        return {
            "latent_dim": state_dict["prior.flow.base.mean"].shape[-1],
            "n_transforms": max(
                int(k.split(".")[3])
                for k in state_dict.keys()
                if k.startswith("prior.flow.transformations")
            ) + 1,
        }
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot prior and aggregate posterior for a 2D-latent VAE."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model .pt file")
    parser.add_argument("--prior", type=str, required=True,
                        choices=["gaussian", "mix", "flow"],
                        help="Prior type used during training")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grid-lim", type=float, default=4.0,
                        help="Grid spans [-lim, lim] on each axis")
    parser.add_argument("--grid-res", type=int, default=200,
                        help="Number of grid points per axis")
    parser.add_argument("--output", type=str, default="prior_vs_posterior.png")
    args = parser.parse_args()

    device = args.device

    # --- Load model ---
    state_dict = torch.load(args.model, map_location=device)
    param_dict = infer_params_from_state_dict(state_dict, args.prior)

    model_args = argparse.Namespace(
        prior=args.prior, latent_dim=param_dict["latent_dim"]
    )
    model = build_model(args=model_args, param_dict=param_dict, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Load data ---
    threshold = 0.5
    mnist_train = datasets.MNIST(
        "data/", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (threshold < x).float().squeeze()),
        ]),
    )
    loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=False)

    # --- Grid ---
    grid_z1 = np.linspace(-args.grid_lim, args.grid_lim, args.grid_res)
    grid_z2 = np.linspace(-args.grid_lim, args.grid_lim, args.grid_res)

    # --- Compute densities ---
    print("Computing prior density...")
    zz1_prior, zz2_prior, prior_density = compute_prior_density(
        model, device, grid_z1, grid_z2
    )

    print("Computing aggregate posterior density (this may take a moment)...")
    zz1_agg, zz2_agg, agg_density = compute_aggregate_posterior_density(
        model, loader, device, grid_z1, grid_z2
    )

    # --- Plot ---
    prior_labels = {
        "gaussian": "Prior",
        "mix": "Prior",
        "flow": "Prior",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].contourf(zz1_prior, zz2_prior, prior_density,
                     levels=50, cmap="viridis")
    #axes[0].set_title(prior_labels[args.prior], fontsize=14)
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    axes[1].contourf(zz1_agg, zz2_agg, agg_density,
                     levels=50, cmap="viridis")
    #axes[1].set_title("Aggregate Posterior", fontsize=14)
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()