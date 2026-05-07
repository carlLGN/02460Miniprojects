# 02460Miniprojects
Codebase for Miniprojects 1, 2 and 3 in course 02460 Advanced Machine Learning at the Technical University of Denmark.

## UV as package manager
Install guide (use standalone installer): https://docs.astral.sh/uv/getting-started/installation/

Each project contains a UV project for the packages that are required for that specific project. To run any code in a project, simply `uv sync --package <project1, project2, project3>` in the root or `uv sync --all-packages` to get all packages from all projects.

Add a package via `uv add --package <project1, project2, project3> <package_name>`.

Invoke a python file via `uv run python <file>.py`. (Not consistent with relative imports)

Alternatively invoke a python file using `uv run python -m <path>` (example: `uv run python -m Project1.src.vae.vae`).


## Project 1: Variational Autoencoders and Diffusion Models                            
                                                                
This project trains and evaluates VAE and DDPM models on MNIST in two parts.
                                                                
### Part A: Priors for VAEs                                                            
                                                                
Trains VAEs with a Bernoulli decoder on binarized MNIST, comparing three prior distributions:  
                                                                
- **Gaussian prior** — Standard N(0, I), closed-form KL divergence                     
- **Mixture of Gaussians (MoG)** — K learnable Gaussian components (K swept over
{1..11})                                                                               
- **Flow-based prior** — Masked affine coupling layers on a Gaussian base (n_transforms
swept over {2, 4, 8, 16, 32})                                                         

The encoder/decoder are MLP-based (784 → 512 → 512 → latent). Models are evaluated on test set ELBO which approximates the true loss with hyperparameter sweeps over latent dimension and learning rate. 
`plot_prior_posterior.py` visualizes the prior density against the 
aggregate posterior in 2D latent space.

### Part B: Sampling Quality of Diffusion Models

Trains diffusion models on standard MNIST (normalized to [-1, 1]) and evaluates using FID scores. Two model types 
are compared:                                                                          

- **Standard DDPM** — Denoising diffusion probabilistic model (T=100 steps) using a    
U-Net backbone with skip connections and SiLU activations
- **Latent DDPM** — DDPM operating in the latent space of a β-VAE (β ∈ {1, 1e-3, 1e-6}), using a fully-connected noise network; samples are decoded back to pixel space via the frozen VAE decoder
                                                                
FID is computed using features from a pre-trained MNIST classifier. `plot_latents.py` compares the prior, aggregate posterior, and latent DDPM distributions via PCA projection.                                                      

### Running Project 1                                                                  

```bash                                                                                
uv sync --package project1
uv run python -m Project1.src.vae.vae          # Train VAE (Part A)
uv run python -m Project1.src.PartB.DDPM       # Train DDPM (Part B)                   
uv run python -m Project1.src.PartB.Latent_DDPM  # Train Latent DDPM (Part B)

```

## Project 2:

## Project 3:                        

