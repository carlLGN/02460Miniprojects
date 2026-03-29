# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn


from Project2.src.vae import *


class EnsembleVAE(nn.Module):
    def __init__(self, prior, decoders, encoder):
        """
        Parameters:
        prior: GaussianPrior instance
        decoders: nn.ModuleList of GaussianDecoder instances
        encoder: GaussianEncoder instance
        """
        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoders = decoders # This should be an nn.ModuleList

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        
        # calculate average loglikelihood across decoders
        logz = 0
        for decoder in self.decoders:
            logz += decoder(z).log_prob(x)
        logz /= len(self.decoders)
        
        elbo = torch.mean(
            logz - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1, decoder_index=None):
        """
        Sample from the model. You can specify a decoder index 
        or it will default to the first one.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        idx = decoder_index if decoder_index is not None else 0
        return self.decoders[idx](z).sample()

    def forward(self, x):
        return -self.elbo(x)
