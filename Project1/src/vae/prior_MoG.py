import torch
import torch.nn as nn
import torch.distributions as td
from torch.nn import functional as F

import pdb


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
            Number of mixture components.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K

        self.pi = nn.Parameter(0.01 * torch.randn(self.K))
        self.mean = nn.Parameter(0.01 * torch.randn(self.K, self.M))

        init_pre_std = torch.log(torch.exp(torch.tensor(1.0)) - 1.0)
        self.pre_stds = nn.Parameter(
        init_pre_std + 0.01 * torch.randn(self.K, self.M)
        )   

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """

        pi_dist = td.Categorical(logits=self.pi)
        stds = F.softplus(self.pre_stds)

        component_dist = td.Independent(td.Normal(loc=self.mean, scale=stds), 1)
        return td.MixtureSameFamily(
                    mixture_distribution=pi_dist, 
                    component_distribution=component_dist
                )
    