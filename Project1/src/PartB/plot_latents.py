import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# --- Import your architectures ---
from Project1.src.PartB.BetaVAE import Beta_VAE, GaussianEncoder, GaussianDecoder
from Project1.src.vae.prior_gaussian import GaussianPrior
from Project1.src.PartB.Latent_DDPM import Latent_DDPM, FcNetwork

def get_real_images(n_samples=2000, device="cpu"):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    x_real = torch.stack([test_ds[i][0] for i in range(n_samples)]).to(device)
    return x_real

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 32
    N = 2000 # Number of points to plot
    
    # --- 1. FILL IN YOUR PATHS HERE ---
    # I highly recommend doing this for BOTH beta=1 and beta=1e-6 and making two plots!
    VAE_PATH = "Project1/src/PartB/beta_vae_model_earlystop_beta1e-6.pt"
    DDPM_PATH = "Project1/src/PartB/latent_ddpm_model_beta1e-6.pt"
    PLOT_TITLE = "Latent Space Comparison (Beta = 1e-6)"
    SAVE_NAME = "latent_plot_beta1e-6.png"

    # --- 2. Setup Models ---
    x_real = get_real_images(N, device)
    
    prior = GaussianPrior(M)
    encoder_net = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(784, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, M * 2)
    )
    decoder_net = torch.nn.Sequential(
        torch.nn.Linear(M, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512),
        torch.nn.ReLU(), torch.nn.Linear(512, 784), torch.nn.Unflatten(-1, (28, 28))
    )
    vae = Beta_VAE(prior, GaussianDecoder(decoder_net), GaussianEncoder(encoder_net)).to(device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()

    latent_ddpm = Latent_DDPM(FcNetwork(M, 512), T=1000).to(device)
    latent_ddpm.load_state_dict(torch.load(DDPM_PATH, map_location=device))
    latent_ddpm.eval()

    # --- 3. Extract the Three Distributions ---
    print("Generating distributions...")
    with torch.no_grad():
        # A. The Prior: Just standard Gaussian noise N(0, I)
        z_prior = torch.randn(N, M).cpu().numpy()
        
        # B. Aggregate Posterior: Real images passed through the VAE encoder
        z_posterior = vae.encoder(x_real).rsample().cpu().numpy()
        
        # C. Learned Latent DDPM: Sampled from the DDPM (NO SCALING)
        z_ddpm = latent_ddpm.sample((N, M)).cpu().numpy()

    # --- 4. PCA Dimensionality Reduction ---
    print("Fitting PCA...")
    pca = PCA(n_components=2)
    # We fit PCA on the Aggregate Posterior because it's the "ground truth" shape
    pca.fit(z_posterior) 

    prior_2d = pca.transform(z_prior)
    post_2d = pca.transform(z_posterior)
    ddpm_2d = pca.transform(z_ddpm)

    # --- 5. Plotting ---
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Plot Prior (Blue)
    plt.scatter(prior_2d[:, 0], prior_2d[:, 1], alpha=0.5, label="Prior N(0,I)", color="royalblue", s=15)
    
    # Plot Aggregate Posterior (Green)
    plt.scatter(post_2d[:, 0], post_2d[:, 1], alpha=0.5, label="Aggregate Posterior", color="forestgreen", s=15)
    
    # Plot DDPM Samples (Red)
    plt.scatter(ddpm_2d[:, 0], ddpm_2d[:, 1], alpha=0.5, label="Latent DDPM", color="crimson", s=15)
    
    plt.title(PLOT_TITLE, fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {SAVE_NAME}")

if __name__ == "__main__":
    main()