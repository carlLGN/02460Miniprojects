# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
# Updated by group 50 for 02460 mini-project 1 spring 2026.

from xml.parsers.expat import model

from pyexpat import model

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

from Project1.src.PartB.GausEncoder_Decoder import GaussianEncoder, GaussianDecoder
from Project1.src.vae.prior_gaussian import GaussianPrior
from Project1.src.vae.prior_MoG import MoGPrior
from Project1.src.vae.prior_flow import FlowPrior

import pdb

from Project1.src.vae.vae import VAE

#TODO: hydra config til at definere prior
#TODO: Andre masking strategier i flow prior?
#TODO: Sampling af flow prior kræver lidt anderledes kode
#TODO: factor out z in flow prior?


class Beta_VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, beta = 1):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(Beta_VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        # If kl_divergence cannot be calculated on closed form, use montecarlo to estimate
        try:
            kl = td.kl_divergence(q, self.prior())
            recon = self.decoder(z).log_prob(x)
            elbo = torch.mean(recon - self.beta * kl, dim=0)
        except NotImplementedError:
            kl = (q.log_prob(z) - self.prior().log_prob(z)).mean(0)
            recon = self.decoder(z).log_prob(x)
            elbo = torch.mean(recon - self.beta * kl, dim = 0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='beta_vae_model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='beta_vae_samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', help='prior for vae, choices are: gaussian, mix, flow (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0   # safe for Windows
    )

    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root='data/',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Define prior distribution
    M = args.latent_dim

    prior = GaussianPrior(M)
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (1, 28, 28))   
    )

    # Define VAE model
    decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    beta_vae_model = Beta_VAE(prior, decoder, encoder).to(device)

    from pathlib import Path

    # Define directory once
    save_dir = Path("Project1/src/PartB")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define model file
    model_path = save_dir / args.model

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(beta_vae_model.parameters(), lr=1e-3)

        # Train model
        train(beta_vae_model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(beta_vae_model.state_dict(), model_path)

    elif args.mode == 'sample':
        beta_vae_model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))

        # Generate samples
        beta_vae_model.eval()
        with torch.no_grad():
            samples = (beta_vae_model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
