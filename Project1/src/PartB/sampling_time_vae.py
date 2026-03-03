import torch
import numpy as np
from torchvision import datasets, transforms
import time
# --- Import your provided FID function ---
from Project1.src.PartB.fid import compute_fid

# --- Import your architectures ---
from Project1.src.PartB.BetaVAE import Beta_VAE, GaussianEncoder, GaussianDecoder
from Project1.src.vae.prior_gaussian import GaussianPrior
from Project1.src.PartB.DDPM import DDPM
from Project1.src.PartB.Latent_DDPM import Latent_DDPM, FcNetwork 
# Don't forget to import your Unet for the standard DDPM!
from Project1.src.PartB.UnetClass import Unet 

def benchmark_model(name, generation_func, N, device):
    """Helper function to warm up and time a generation function."""
    print(f"\n--- Benchmarking {name} ---")
    
    with torch.no_grad():
        # 1. Warm-up (run a tiny batch of 2 to initialize memory)
        print("  Warming up...")
        _ = generation_func(2)
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        # 2. Time the actual generation
        print(f"  Generating {N} samples...")
        start_time = time.perf_counter()
        
        _ = generation_func(N)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()

    total_time = end_time - start_time
    time_per_sample = total_time / N
    
    print(f"  Done in {total_time:.4f} seconds.")
    return total_time, time_per_sample

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    N = 100  # Number of samples to generate per model
    M = 32   # Latent dimension
    
    # ==========================================
    # 1. FILL IN YOUR PATHS HERE
    # ==========================================
    VAE_PATH = "Project1/src/PartB/beta_vae_model_earlystop_beta1e-6.pt"
    STANDARD_DDPM_PATH = "Project1/src/PartB/ddpm_earlystop.pt"
    LATENT_DDPM_PATH = "Project1/src/PartB/latent_ddpm_model_beta1e-6.pt"

    # ==========================================
    # 2. Setup Architectures
    # ==========================================
    # VAE
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

    # Standard DDPM
    standard_ddpm = DDPM(Unet(), T=1000).to(device)
    standard_ddpm.load_state_dict(torch.load(STANDARD_DDPM_PATH, map_location=device))
    standard_ddpm.eval()

    # Latent DDPM
    latent_ddpm = Latent_DDPM(FcNetwork(M, 512), T=1000).to(device)
    latent_ddpm.load_state_dict(torch.load(LATENT_DDPM_PATH, map_location=device))
    latent_ddpm.eval()

    # ==========================================
    # 3. Define Generation Functions
    # ==========================================
    # We wrap the generation calls in lambda functions so the benchmark helper 
    # can just pass in the batch size 'n' and execute them.

    # VAE: Just sample from the prior and decode
    gen_vae = lambda n: vae.sample(n)
    
    # Standard DDPM: Sample flattened, then reshape (handling the Unet quirk)
    gen_standard_ddpm = lambda n: standard_ddpm.sample((n, 784)).view(n, 1, 28, 28)
    
    # Latent DDPM: Sample latents, then decode through VAE
    gen_latent_ddpm = lambda n: vae.decoder(latent_ddpm.sample((n, M))).mean

    # ==========================================
    # 4. Run Benchmarks
    # ==========================================
    results = {}
    results["Beta-VAE"] = benchmark_model("Beta-VAE", gen_vae, N, device)
    results["Latent DDPM"] = benchmark_model("Latent DDPM", gen_latent_ddpm, N, device)
    results["Standard DDPM"] = benchmark_model("Standard DDPM", gen_standard_ddpm, N, device)

    # ==========================================
    # 5. Print Final Report
    # ==========================================
    print("\n" + "="*50)
    print("FINAL TIMING REPORT".center(50))
    print("="*50)
    print(f"{'Model':<18} | {'Total Time (s)':<14} | {'Time per Sample (s)':<14}")
    print("-" * 50)
    for name, (total, per_sample) in results.items():
        print(f"{name:<18} | {total:<14.4f} | {per_sample:<14.6f}")
    print("="*50)

if __name__ == "__main__":
    main()