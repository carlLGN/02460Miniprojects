# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
# Updated by group 50 for 02460 mini-project 1 spring 2026.

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

from Project1.src.vae.encoder_decoder import GaussianEncoder, BernoulliDecoder
from Project1.src.vae.prior_gaussian import GaussianPrior
from Project1.src.vae.prior_MoG import MoGPrior
from Project1.src.vae.prior_flow import FlowPrior

import pdb

#TODO: hydra config til at definere prior
#TODO: Andre masking strategier i flow prior?
#TODO: Sampling af flow prior kræver lidt anderledes kode
#TODO: factor out z in flow prior?


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

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
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        except NotImplementedError:
            kl = (q.log_prob(z) - self.prior().log_prob(z)).mean(0)
            elbo = torch.mean(self.decoder(z).log_prob(x) - kl, dim = 0)
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


def train(model, optimizer, data_loader, val_data_loader, epochs, device):
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
    best_model = model

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    curr_epoch_loss = -torch.inf
    tolerance = 2
    curr_fails = 0

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
    
        validation_loss = validate(model, val_data_loader=val_data_loader, device=device)
        print(f"Epoch loss: {validation_loss}")
        if validation_loss < curr_epoch_loss:
            curr_fails += 1
            if curr_fails >= tolerance:
                print(f"Early stopping at epoch {epoch}.")
                break
            curr_epoch_loss = validation_loss
        else:
            best_model = model
            curr_epoch_loss = validation_loss
            curr_fails = 0

    return best_model


def validate(model, val_data_loader, device):

    model.eval()

    data_iter = iter(val_data_loader)
    test_elbo = torch.ones(len(data_iter))*torch.inf
    for k, x in enumerate(data_iter):
        batch_elbo = model.elbo(x[0].to(device))
        test_elbo[k] = batch_elbo

    model.train()

    return torch.mean(test_elbo)


def test(model, data_loader, device, model_path):

    model.eval()

    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    data_iter = iter(data_loader)
    test_elbo = torch.ones(len(data_iter))*torch.inf
    for k, x in enumerate(data_iter):
        batch_elbo = model.elbo(x[0].to(device))
        test_elbo[k] = batch_elbo
    return torch.mean(test_elbo)


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', help='prior for vae, choices are: gaussian, mix, flow (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    

    mnist_train_dataset = datasets.MNIST('data/', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())]))

    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                batch_size=args.batch_size, shuffle=True)

    train_subset, val_subset = torch.utils.data.random_split(mnist_train_dataset, [0.9, 0.1])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Define prior distribution
    M = args.latent_dim

    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mix':
        prior = MoGPrior(M, K = 5)
    elif args.prior == 'flow':
        prior = FlowPrior(M, n_transformations=4)
    else:
        raise NotImplementedError('Prior does not exist')

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
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        best_model = train(model, optimizer, train_loader, val_loader, args.epochs, args.device)

        # Save model
        torch.save(best_model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'test':
        elbo_loss = test(model, mnist_test_loader, args.device, args.model)
        print(f"Elbo loss for {args.model}: {elbo_loss}")

