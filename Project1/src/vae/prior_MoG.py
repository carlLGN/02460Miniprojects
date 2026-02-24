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

        self.pi = nn.Parameter(torch.zeros(self.K), requires_grad=True) #TODO: Skal den learnes or no ?
        self.mean = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=True)
        self.pre_stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)

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