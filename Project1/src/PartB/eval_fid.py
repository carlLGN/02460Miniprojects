import torch
import numpy as np
from torchvision import datasets, transforms

# --- Import your provided FID function ---
from Project1.src.PartB.fid import compute_fid

# --- Import your architectures ---
from Project1.src.PartB.BetaVAE import Beta_VAE, GaussianEncoder, GaussianDecoder
from Project1.src.vae.prior_gaussian import GaussianPrior
from Project1.src.PartB.DDPM import DDPM
from Project1.src.PartB.Latent_DDPM import Latent_DDPM, FcNetwork 
# Don't forget to import your Unet for the standard DDPM!
from Project1.src.PartB.UnetClass import Unet 


def get_real_images(n_samples=1000, device="cpu"):
    """Fetch real MNIST images in [-1, 1] range."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    x_real = torch.stack([test_ds[i][0] for i in range(n_samples)]).to(device)
    return x_real

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 1000 # 1000 samples is a standard balance for FID speed/accuracy
    M = 32   # Latent dimension
    
    # ==========================================
    # 1. FILL IN YOUR PATHS HERE
    # ==========================================
    CLASSIFIER_PATH = "Project1/src/PartB/mnist_classifier.pth" # <-- Fixes your error!
    
    STANDARD_DDPM_PATH = "Project1/src/PartB/ddpm_earlystop.pt"
    CHOSEN_VAE_PATH = "Project1/src/PartB/beta_vae_model_earlystop_beta1.pt" # VAE of your choice
    
    # Put your three Latent DDPM paths and their corresponding VAE paths here
    LATENT_RUNS = [
        {
            "name": "Latent DDPM (Beta=1)",
            "ddpm_path": "Project1/src/PartB/latent_ddpm_model_beta1.pt",
            "vae_path": "Project1/src/PartB/beta_vae_model_earlystop_beta1.pt"
        },
        {
            "name": "Latent DDPM (Beta=1e-3)", 
            "ddpm_path": "Project1/src/PartB/latent_ddpm_model_beta1e-3.pt",
            "vae_path": "Project1/src/PartB/beta_vae_model_earlystop_beta1e-3.pt"
        },
        {
            "name": "Latent DDPM (Beta=1e-6)",
            "ddpm_path": "Project1/src/PartB/latent_ddpm_model_beta1e-6.pt",
            "vae_path": "Project1/src/PartB/beta_vae_model_earlystop_beta1e-6.pt"
        }
    ]
    # ==========================================

    print(f"Fetching {N} real images...")
    x_real = get_real_images(n_samples=N, device=device)

    # ==========================================
    # 2. Evaluate Standard DDPM
    # ==========================================
    print("\n--- Evaluating Standard DDPM ---")
    standard_ddpm = DDPM(Unet(), T=1000).to(device)
    standard_ddpm.load_state_dict(torch.load(STANDARD_DDPM_PATH, map_location=device))
    standard_ddpm.eval()
    
    with torch.no_grad():
        # 1. Sample flattened (N, 784)
        x_gen_standard = standard_ddpm.sample((N, 784))
        # 2. Reshape back to images for the FID function
        x_gen_standard = x_gen_standard.view(N, 1, 28, 28).clamp(-1, 1)
        
    fid_standard = compute_fid(x_real, x_gen_standard, device=str(device), classifier_ckpt=CLASSIFIER_PATH)
    print(f"Standard DDPM FID: {fid_standard:.4f}")

    # ==========================================
    # 3. Setup VAE Architecture (Reused below)
    # ==========================================
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

    # ==========================================
    # 4. Evaluate Chosen VAE
    # ==========================================
    print("\n--- Evaluating Chosen Beta-VAE ---")
    vae.load_state_dict(torch.load(CHOSEN_VAE_PATH, map_location=device))
    vae.eval()
    
    with torch.no_grad():
        z_vae = torch.randn(N, M, device=device)
        x_gen_vae = vae.decoder(z_vae).mean.unsqueeze(1).clamp(-1, 1)
        
    fid_vae = compute_fid(x_real, x_gen_vae, device=str(device), classifier_ckpt=CLASSIFIER_PATH)
    print(f"Chosen VAE FID: {fid_vae:.4f}")

    # ==========================================
    # 5. Evaluate All Latent DDPMs
    # ==========================================
    latent_ddpm = Latent_DDPM(FcNetwork(M, 512), T=1000).to(device)

    for run in LATENT_RUNS:
        print(f"\n--- Evaluating {run['name']} ---")
        
        # Load the specific VAE for this run
        vae.load_state_dict(torch.load(run['vae_path'], map_location=device))
        vae.eval()
        
        # Load the specific Latent DDPM for this run
        latent_ddpm.load_state_dict(torch.load(run['ddpm_path'], map_location=device))
        latent_ddpm.eval()

        with torch.no_grad():
            # Sample latents (NO SCALING APPLIED)
            z_ddpm = latent_ddpm.sample((N, M))
            
            # Decode to pixels
            x_gen_latent = vae.decoder(z_ddpm).mean.unsqueeze(1).clamp(-1, 1)

        fid_latent = compute_fid(x_real, x_gen_latent, device=str(device), classifier_ckpt=CLASSIFIER_PATH)
        print(f"{run['name']} FID: {fid_latent:.4f}")

if __name__ == "__main__":
    main()